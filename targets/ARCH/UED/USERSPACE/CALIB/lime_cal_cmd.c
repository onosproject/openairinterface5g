#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>

#include "openair0_lib.h"

#include "syr_pio.h"
#include "lime_cal_cmd.h"
#include "lime_spi_cmd.h"
#include "lime_reg_cmd.h"

#define INC_BUSY_WAIT_COUNT(us) do {} while (0)
#define RESET_BUSY_WAIT_COUNT() do {} while (0)
#define PRINT_BUSY_WAIT_INFO()
#define VTUNE_BUSY_WAIT(us) { usleep(us); INC_BUSY_WAIT_COUNT(us); }

#define LMS_REFERENCE_HZ    (30720000u)

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

extern	void	*cmdcontext_spi;

uint32_t lms_frequency_to_hz(struct lms_freq *f)
{
    uint64_t pll_coeff;
    uint32_t div;

    pll_coeff = (((uint64_t)f->nint) << 23) + f->nfrac;
    div = (f->x << 23);

    return (uint32_t)(((LMS_REFERENCE_HZ * pll_coeff) + (div >> 1)) / div);
}

void lms_print_frequency(struct lms_freq *f)
{
    printf("---- Frequency ----\r\n");
    printf("  x        : %d\r\n", f->x);
    printf("  nint     : %d\r\n", f->nint);
    printf("  nfrac    : %u\r\n", f->nfrac);
    printf("  freqsel  : 0x%02x\r\n", f->freqsel);
    printf("  reference: %u\r\n", LMS_REFERENCE_HZ);
    printf("  freq     : %u\r\n", lms_frequency_to_hz(f));
}
#define PRINT_FREQUENCY lms_print_frequency

/* cf. BladeRF */
int lms_soft_reset(void *cmdcontext_spi)
{
	unsigned char	reg0x05_val		= 0;

	lime_spi_blk_write(cmdcontext_spi,	0x05, 0x12);
	lime_spi_blk_read(cmdcontext_spi,	0x05, &reg0x05_val);
	if (reg0x05_val == 0x12)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x05, 0x32);
		lime_spi_blk_read(cmdcontext_spi,	0x05, &reg0x05_val);
		if (reg0x05_val == 0x32)
		{
			return 0;	// OK
		}
	}
    return (-1);		// NOK
}

int lms_enable_rffe(void *dev, bladerf_module module, bool enable)
{
	int		status;
	uint8_t	data;
	uint8_t	addr	= (module == BLADERF_MODULE_TX ? 0x40 : 0x70);

	status = LMS_READ(dev, addr, &data);
	if (status == 0)
	{
		if (module == BLADERF_MODULE_TX)
		{
			if (enable)
			{
				data |= (1 << 1);
			}
			else
			{
				data &= ~(1 << 1);
			}
		}
		else
		{
			if (enable)
			{
				data |= (1 << 0);
			}
			else
			{
				data &= ~(1 << 0);
			}
		}
		status = LMS_WRITE(dev, addr, data);
	}
	return status;
}

int lms_config_charge_pumps(void *dev, bladerf_module module)
{
	int				status;
	uint8_t			data;
	const uint8_t	base	= (module == BLADERF_MODULE_RX) ? 0x20 : 0x10;

	/* Set the PLL Ichp, Iup and Idn currents */
	status = LMS_READ(dev, base + 6, &data);	// 0x26 | 0x16
	if (status != 0)
	{
		return status;
	}
	data &= ~(0x1f);
	data |= 0x0c;
	status = LMS_WRITE(dev, base + 6, data);	/* 0x26 | 0x16 = 0bxxx01100 : Charge pump current=1200µA */
	if (status != 0)
	{
		return status;
	}

	status = LMS_READ(dev, base + 7, &data);	// 0x27 | 0x17
	if (status != 0)
	{
		return status;
	}
	data &= ~(0x1f);
	data |= 3;
	status = LMS_WRITE(dev, base + 7, data);	/* 0x27 | 0x17 = 0bxxx00011 : Charge pump UP offset current=30µA */
	if (status != 0)
	{
		return status;
	}

	status = LMS_READ(dev, base + 8, &data);	// 0x28 | 0x18
	if (status != 0)
	{
		return status;
	}
	data &= ~(0x1f);
	data |= 0;
	status = LMS_WRITE(dev, base + 8, data);	/* 0x28 | 0x18 = 0bxxx00000 : Charge pump DOWN offset current=0µA */
	if (status != 0)
	{
		return status;
	}
	return 0;
}

/* For >= 1.5 GHz uses the high band should be used. Otherwise, the low
 * band should be selected */
#define BLADERF_BAND_HIGH (1500000000)

/* Just for easy copy/paste back into lms.c code */
#define FREQ_RANGE(l, h, v) {l, h, v}

struct freq_range {
    uint32_t    low;
    uint32_t    high;
    uint8_t     value;
};

struct vco_range {
    uint64_t low;
    uint64_t high;
};

/* It appears that VCO1-4 are mislabeled in the LMS6002D FAQ 5.24 plot.
 * Note that the below are labeled "correctly," so they will not match the plot.
 *
 * Additionally, these have been defined a bit more conservatively, rounding
 * to the nearest 100 MHz contained within a band.
 */
static const struct vco_range vco[] = {
    { 0,                       0},            /* Dummy entry */
    { 6300000000u,             7700000000u }, /* VCO1 */
    { 5300000000u,             6900000000u },
    { 4500000000u,             5600000000u },
    { 3700000000u,             4700000000u }, /* VCO4 */
};

/** Minimum tunable frequency (without an XB-200 attached), in Hz */
#define BLADERF_FREQUENCY_MIN       237500000ull

/** Maximum tunable frequency, in Hz */
#define BLADERF_FREQUENCY_MAX       3800000000ull

/* Here we define more conservative band ranges than those in the
 * LMS FAQ (5.24), with the intent of avoiding the use of "edges" that might
 * cause the PLLs to lose lock over temperature changes */
#define VCO4_LOW    3800000000ull
#define VCO4_HIGH   4535000000ull

#define VCO3_LOW    VCO4_HIGH
#define VCO3_HIGH   5408000000ull

#define VCO2_LOW    VCO3_HIGH
#define VCO2_HIGH   6480000000ull

#define VCO1_LOW    VCO2_HIGH
#define VCO1_HIGH   7600000000ull

/* SELVCO values */
#define VCO4 (4 << 3)
#define VCO3 (5 << 3)
#define VCO2 (6 << 3)
#define VCO1 (7 << 3)

/* FRANGE values */
#define DIV2  0x4
#define DIV4  0x5
#define DIV8  0x6
#define DIV16 0x7

/* Additional changes made after tightening "keepout" percentage */
static const struct freq_range bands[] = {
    FREQ_RANGE(BLADERF_FREQUENCY_MIN,   VCO4_HIGH/16,           VCO4 | DIV16),
    FREQ_RANGE(VCO3_LOW/16,             VCO3_HIGH/16,           VCO3 | DIV16),
    FREQ_RANGE(VCO2_LOW/16,             VCO2_HIGH/16,           VCO2 | DIV16),
    FREQ_RANGE(VCO1_LOW/16,             VCO1_HIGH/16,           VCO1 | DIV16),
    FREQ_RANGE(VCO4_LOW/8,              VCO4_HIGH/8,            VCO4 | DIV8),
    FREQ_RANGE(VCO3_LOW/8,              VCO3_HIGH/8,            VCO3 | DIV8),
    FREQ_RANGE(VCO2_LOW/8,              VCO2_HIGH/8,            VCO2 | DIV8),
    FREQ_RANGE(VCO1_LOW/8,              VCO1_HIGH/8,            VCO1 | DIV8),
    FREQ_RANGE(VCO4_LOW/4,              VCO4_HIGH/4,            VCO4 | DIV4),
    FREQ_RANGE(VCO3_LOW/4,              VCO3_HIGH/4,            VCO3 | DIV4),
    FREQ_RANGE(VCO2_LOW/4,              VCO2_HIGH/4,            VCO2 | DIV4),
    FREQ_RANGE(VCO1_LOW/4,              VCO1_HIGH/4,            VCO1 | DIV4),
    FREQ_RANGE(VCO4_LOW/2,              VCO4_HIGH/2,            VCO4 | DIV2),
    FREQ_RANGE(VCO3_LOW/2,              VCO3_HIGH/2,            VCO3 | DIV2),
    FREQ_RANGE(VCO2_LOW/2,              VCO2_HIGH/2,            VCO2 | DIV2),
    FREQ_RANGE(VCO1_LOW/2,              BLADERF_FREQUENCY_MAX,  VCO1 | DIV2),
};

static const size_t num_bands = sizeof(bands) / sizeof(bands[0]);

#define kHz(x) (x * 1000)
#define MHz(x) (x * 1000000)
#define GHz(x) (x * 1000000000)

/**
 * @defgroup    RETCODES    Error codes
 *
 * bladeRF library routines return negative values to indicate errors.
 * Values >= 0 are used to indicate success.
 *
 * @code
 *  int status = bladerf_set_txvga1(dev, 2);
 *
 *  if (status < 0)
 *      handle_error();
 * @endcode
 *
 * @{
 */
#define BLADERF_ERR_UNEXPECTED  (-1)  /**< An unexpected failure occurred */
#define BLADERF_ERR_RANGE       (-2)  /**< Provided parameter is out of range */
#define BLADERF_ERR_INVAL       (-3)  /**< Invalid operation/parameter */
#define BLADERF_ERR_MEM         (-4)  /**< Memory allocation error */
#define BLADERF_ERR_IO          (-5)  /**< File/Device I/O error */
#define BLADERF_ERR_TIMEOUT     (-6)  /**< Operation timed out */
#define BLADERF_ERR_NODEV       (-7)  /**< No device(s) available */
#define BLADERF_ERR_UNSUPPORTED (-8)  /**< Operation not supported */
#define BLADERF_ERR_MISALIGNED  (-9)  /**< Misaligned flash access */
#define BLADERF_ERR_CHECKSUM    (-10) /**< Invalid checksum */
#define BLADERF_ERR_NO_FILE     (-11) /**< File not found */
#define BLADERF_ERR_UPDATE_FPGA (-12) /**< An FPGA update is required */
#define BLADERF_ERR_UPDATE_FW   (-13) /**< A firmware update is requied */
#define BLADERF_ERR_TIME_PAST   (-14) /**< Requested timestamp is in the past */
#define BLADERF_ERR_QUEUE_FULL  (-15) /**< Could not enqueue data into
                                       *   full queue */
#define BLADERF_ERR_FPGA_OP     (-16) /**< An FPGA operation reported failure */
#define BLADERF_ERR_PERMISSION  (-17) /**< Insufficient permissions for the
                                       *   requested operation */
#define BLADERF_ERR_WOULD_BLOCK (-18) /**< Operation would block, but has been
                                       *   requested to be non-blocking. This
                                       *   indicates to a caller that it may
                                       *   need to retry the operation later.
                                       */

/* VCOCAP estimation. The MIN/MAX values were determined experimentally by
 * sampling the VCOCAP values over frequency, for each of the VCOs and finding
 * these to be in the "middle" of a linear regression. Although the curve
 * isn't actually linear, the linear approximation yields satisfactory error. */
#define VCOCAP_MAX_VALUE 0x3f
#define VCOCAP_EST_MIN 15
#define VCOCAP_EST_MAX 55
#define VCOCAP_EST_RANGE (VCOCAP_EST_MAX - VCOCAP_EST_MIN)
#define VCOCAP_EST_THRESH 7 /* Complain if we're +/- 7 on our guess */

/* This is a linear interpolation of our experimentally identified
 * mean VCOCAP min and VCOCAP max values:
 */
static inline uint8_t estimate_vcocap(unsigned int f_target,
                                      unsigned int f_low, unsigned int f_high)
{
    unsigned int vcocap;
    const float denom = (float) (f_high - f_low);
    const float num = VCOCAP_EST_RANGE;
    const float f_diff = (float) (f_target - f_low);

    vcocap = (unsigned int) ((num / denom * f_diff) + 0.5 + VCOCAP_EST_MIN);

    if (vcocap > VCOCAP_MAX_VALUE) {
        printf("Clamping VCOCAP estimate from %u to %u\r\n",
                    vcocap, VCOCAP_MAX_VALUE);
        vcocap = VCOCAP_MAX_VALUE;
    } else {
    	printf("VCOCAP estimate: %u\r\n", vcocap);
    }
    return (uint8_t) vcocap;
}

int lms_calculate_tuning_params(uint32_t freq, struct lms_freq *f)
{
    uint64_t vco_x;
    uint64_t temp;
    uint16_t nint;
    uint32_t nfrac;
    uint8_t freqsel = bands[0].value;
    uint8_t i = 0;
    const uint64_t ref_clock = LMS_REFERENCE_HZ;

	printf("lms_calculate_tuning_params(uint32_t freq, struct lms_freq *f)\r\n");

    /* Clamp out of range values */
    if (freq < BLADERF_FREQUENCY_MIN) {
        freq = BLADERF_FREQUENCY_MIN;
        printf("Clamping frequency to %uHz\r\n", freq);
    } else if (freq > BLADERF_FREQUENCY_MAX) {
        freq = BLADERF_FREQUENCY_MAX;
        printf("Clamping frequency to %uHz\r\n", freq);
    }

    /* Figure out freqsel */

    while (i < ARRAY_SIZE(bands)) {
        if ((freq >= bands[i].low) && (freq <= bands[i].high)) {
            freqsel = bands[i].value;
            break;
        }
        i++;
    }

    /* This condition should never occur. There's a bug if it does. */
    if (i >= ARRAY_SIZE(bands)) {
        printf("BUG: Failed to find frequency band information. Setting frequency to %llu Hz.\r\n", BLADERF_FREQUENCY_MIN);

        return BLADERF_ERR_UNEXPECTED;
    }

    /* Estimate our target VCOCAP value. */
    f->vcocap = estimate_vcocap(freq, bands[i].low, bands[i].high);

    /* Calculate integer portion of the frequency value */
    vco_x = ((uint64_t)1) << ((freqsel & 7) - 3);
    temp = (vco_x * freq) / ref_clock;
    //assert(temp <= UINT16_MAX);
    nint = (uint16_t)temp;

    temp = (1 << 23) * (vco_x * freq - nint * ref_clock);
    temp = (temp + ref_clock / 2) / ref_clock;
    //assert(temp <= UINT32_MAX);
    nfrac = (uint32_t)temp;

    //assert(vco_x <= UINT8_MAX);
    f->x = (uint8_t)vco_x;
    f->nint = nint;
    f->nfrac = nfrac;
    f->freqsel = freqsel;
    //assert(ref_clock <= UINT32_MAX);

    f->flags = 0;

    if (freq < BLADERF_BAND_HIGH) {
        f->flags |= LMS_FREQ_FLAGS_LOW_BAND;
    }

    PRINT_FREQUENCY(f);

    return 0;
}

static inline int get_vtune(void *dev, uint8_t base, uint8_t delay,
                            uint8_t *vtune)
{
    int status;

    if (delay != 0) {
        VTUNE_BUSY_WAIT(delay);
    }

    status = LMS_READ(dev, base + 10, vtune);
    *vtune >>= 6;

    return status;
}

static inline int write_vcocap(void *dev, uint8_t base,
                               uint8_t vcocap, uint8_t vcocap_reg_state)
{
    int status;

    //assert(vcocap <= VCOCAP_MAX_VALUE);
    printf("Writing VCOCAP=%u\r\n", vcocap);

    status = LMS_WRITE(dev, base + 9, vcocap | vcocap_reg_state);

    if (status != 0) {
        printf("VCOCAP write failed: %d\r\n", status);
    }

    return status;
}

#define VTUNE_DELAY_LARGE 50
#define VTUNE_DELAY_SMALL 25
#define VTUNE_MAX_ITERATIONS 20

#define VCO_HIGH 0x02
#define VCO_NORM 0x00
#define VCO_LOW  0x01

static const char *vtune_str(uint8_t value) {
    switch (value) {
        case VCO_HIGH:
            return "HIGH";

        case VCO_NORM:
            return "NORM";

        case VCO_LOW:
            return "LOW";

        default:
            return "INVALID";
    }
}

static int vtune_high_to_norm(void *dev, uint8_t base,
                                     uint8_t vcocap, uint8_t vcocap_reg_state,
                                     uint8_t *vtune_high_limit)
{
    int status;
    unsigned int i;
    uint8_t vtune = 0xff;

    for (i = 0; i < VTUNE_MAX_ITERATIONS; i++) {

        if (vcocap >= VCOCAP_MAX_VALUE) {
            *vtune_high_limit = VCOCAP_MAX_VALUE;
            printf("%s: VCOCAP hit max value.\r\n", __FUNCTION__);
            return 0;
        }

        vcocap++;

        status = write_vcocap(dev, base, vcocap, vcocap_reg_state);
        if (status != 0) {
            return status;
        }

        status = get_vtune(dev, base, VTUNE_DELAY_SMALL, &vtune);
        if (status != 0) {
            return status;
        }

        if (vtune == VCO_NORM) {
            *vtune_high_limit = vcocap - 1;
            printf("VTUNE NORM @ VCOCAP=%u\r\n", vcocap);
            printf("VTUNE HIGH @ VCOCAP=%u\r\n", *vtune_high_limit);
            return 0;
        }
    }

    printf("VTUNE High->Norm loop failed to converge.\r\n");
    return BLADERF_ERR_UNEXPECTED;
}

static int vtune_norm_to_high(void *dev, uint8_t base,
                                     uint8_t vcocap, uint8_t vcocap_reg_state,
                                     uint8_t *vtune_high_limit)
{
    int status;
    unsigned int i;
    uint8_t vtune = 0xff;

    for (i = 0; i < VTUNE_MAX_ITERATIONS; i++) {

        if (vcocap == 0) {
            *vtune_high_limit = 0;
            printf("%s: VCOCAP hit min value.\r\n", __FUNCTION__);
            return 0;
        }

        vcocap--;

        status = write_vcocap(dev, base, vcocap, vcocap_reg_state);
        if (status != 0) {
            return status;
        }

        status = get_vtune(dev, base, VTUNE_DELAY_SMALL, &vtune);
        if (status != 0) {
            return status;
        }

        if (vtune == VCO_HIGH) {
            *vtune_high_limit = vcocap;
            printf("VTUNE high @ VCOCAP=%u\r\n", *vtune_high_limit);
            return 0;
        }
    }

    printf("VTUNE High->Norm loop failed to converge.\r\n");
    return BLADERF_ERR_UNEXPECTED;
}

static int vtune_low_to_norm(void *dev, uint8_t base,
                                    uint8_t vcocap, uint8_t vcocap_reg_state,
                                    uint8_t *vtune_low_limit)
{
    int status;
    unsigned int i;
    uint8_t vtune = 0xff;

    for (i = 0; i < VTUNE_MAX_ITERATIONS; i++) {

        if (vcocap == 0) {
            *vtune_low_limit = 0;
            printf("VCOCAP hit min value.\r\n");
            return 0;
        }

        vcocap--;

        status = write_vcocap(dev, base, vcocap, vcocap_reg_state);
        if (status != 0) {
            return status;
        }

        status = get_vtune(dev, base, VTUNE_DELAY_SMALL, &vtune);
        if (status != 0) {
            return status;
        }

        if (vtune == VCO_NORM) {
            *vtune_low_limit = vcocap + 1;
            printf("VTUNE NORM @ VCOCAP=%u\r\n", vcocap);
            printf("VTUNE LOW @ VCOCAP=%u\r\n", *vtune_low_limit);
            return 0;
        }
    }

    printf("VTUNE Low->Norm loop failed to converge.\r\n");
    return BLADERF_ERR_UNEXPECTED;
}

/* Wait for VTUNE to reach HIGH or LOW. NORM is not a valid option here */
static int wait_for_vtune_value(void *dev,
                                uint8_t base, uint8_t target_value,
                                uint8_t *vcocap, uint8_t vcocap_reg_state)
{
    uint8_t vtune;
    unsigned int i;
    int status = 0;
    const unsigned int max_retries = 15;
    const uint8_t limit = (target_value == VCO_HIGH) ? 0 : VCOCAP_MAX_VALUE;
    int8_t inc = (target_value == VCO_HIGH) ? -1 : 1;

    //assert(target_value == VCO_HIGH || target_value == VCO_LOW);

    for (i = 0; i < max_retries; i++) {
        status = get_vtune(dev, base, 0, &vtune);
        if (status != 0) {
            return status;
        }

        if (vtune == target_value) {
            printf("VTUNE reached %s at iteration %u\r\n",
                        vtune_str(target_value), i);
            return 0;
        } else {
            printf("VTUNE was %s. Waiting and retrying...\r\n",
                        vtune_str(vtune));

            VTUNE_BUSY_WAIT(10);
        }
    }

    printf("Timed out while waiting for VTUNE=%s. Walking VCOCAP...\r\n",
               vtune_str(target_value));

    while (*vcocap != limit) {
        *vcocap += inc;

        status = write_vcocap(dev, base, *vcocap, vcocap_reg_state);
        if (status != 0) {
            return status;
        }

        status = get_vtune(dev, base, VTUNE_DELAY_SMALL, &vtune);
        if (status != 0) {
            return status;
        } else if (vtune == target_value) {
            printf("VTUNE=%s reached with VCOCAP=%u\r\n",
                      vtune_str(vtune), *vcocap);
            return 0;
        }
    }

    printf("VTUNE did not reach %s. Tuning may not be nominal.\r\n",
                vtune_str(target_value));

#   ifdef ERROR_ON_NO_VTUNE_LIMIT
        return BLADERF_ERR_UNEXPECTED;
#   else
        return 0;
#   endif
}

/* These values are the max counts we've seen (experimentally) between
 * VCOCAP values that converged */
#define VCOCAP_MAX_LOW_HIGH  12

/* This function assumes an initial VCOCAP estimate has already been written.
 *
 * Remember, increasing VCOCAP works towards a lower voltage, and vice versa:
 * From experimental observations, we don't expect to see the "normal" region
 * extend beyond 16 counts.
 *
 *  VCOCAP = 0              VCOCAP=63
 * /                                 \
 * v                                  v
 * |----High-----[ Normal ]----Low----|     VTUNE voltage comparison
 *
 * The VTUNE voltage can be found on R263 (RX) or R265 (Tx). (They're under the
 * can shielding the LMS6002D.) By placing a scope probe on these and retuning,
 * you should be able to see the relationship between VCOCAP changes and
 * the voltage changes.
 */
static int tune_vcocap(void *dev, uint8_t vcocap_est,
                       uint8_t base, uint8_t vcocap_reg_state,
                       uint8_t *vcocap_result)
{
    int status;
    uint8_t vcocap = vcocap_est;
    uint8_t vtune;
    uint8_t vtune_high_limit; /* Where VCOCAP puts use into VTUNE HIGH region */
    uint8_t vtune_low_limit;  /* Where VCOCAP puts use into VTUNE HIGH region */

    RESET_BUSY_WAIT_COUNT();

    vtune_high_limit = VCOCAP_MAX_VALUE;
    vtune_low_limit = 0;

    status = get_vtune(dev, base, VTUNE_DELAY_LARGE, &vtune);
    if (status != 0) {
        return status;
    }

    switch (vtune) {
        case VCO_HIGH:
            printf("Estimate HIGH: Walking down to NORM.\r\n");
            status = vtune_high_to_norm(dev, base, vcocap, vcocap_reg_state,
                                        &vtune_high_limit);
            break;

        case VCO_NORM:
            printf("Estimate NORM: Walking up to HIGH.\r\n");
            status = vtune_norm_to_high(dev, base, vcocap, vcocap_reg_state,
                                        &vtune_high_limit);
            break;

        case VCO_LOW:
            printf("Estimate LOW: Walking down to NORM.\r\n");
            status = vtune_low_to_norm(dev, base, vcocap, vcocap_reg_state,
                                       &vtune_low_limit);
            break;
    }

    if (status != 0) {
        return status;
    } else if (vtune_high_limit != VCOCAP_MAX_VALUE) {

        /* We determined our VTUNE HIGH limit. Try to force ourselves to the
         * LOW limit and then walk back up to norm from there.
         *
         * Reminder - There's an inverse relationship between VTUNE and VCOCAP
         */
        switch (vtune) {
            case VCO_HIGH:
            case VCO_NORM:
                if ( ((int) vtune_high_limit + VCOCAP_MAX_LOW_HIGH) < VCOCAP_MAX_VALUE) {
                    vcocap = vtune_high_limit + VCOCAP_MAX_LOW_HIGH;
                } else {
                    vcocap = VCOCAP_MAX_VALUE;
                    printf("Clamping VCOCAP to %u.\r\n", vcocap);
                }
                break;

            default:
                //assert(!"Invalid state");
                return BLADERF_ERR_UNEXPECTED;
        }

        status = write_vcocap(dev, base, vcocap, vcocap_reg_state);
        if (status != 0) {
            return status;
        }

        printf("Waiting for VTUNE LOW @ VCOCAP=%u,\r\n", vcocap);
        status = wait_for_vtune_value(dev, base, VCO_LOW,
                                      &vcocap, vcocap_reg_state);

        if (status == 0) {
            printf("Walking VTUNE LOW to NORM from VCOCAP=%u,\r\n", vcocap);
            status = vtune_low_to_norm(dev, base, vcocap, vcocap_reg_state,
                                       &vtune_low_limit);
        }
    } else {

        /* We determined our VTUNE LOW limit. Try to force ourselves up to
         * the HIGH limit and then walk down to NORM from there
         *
         * Reminder - There's an inverse relationship between VTUNE and VCOCAP
         */
        switch (vtune) {
            case VCO_LOW:
            case VCO_NORM:
                if ( ((int) vtune_low_limit - VCOCAP_MAX_LOW_HIGH) > 0) {
                    vcocap = vtune_low_limit - VCOCAP_MAX_LOW_HIGH;
                } else {
                    vcocap = 0;
                    printf("Clamping VCOCAP to %u.\r\n", vcocap);
                }
                break;

            default:
                //assert(!"Invalid state");
                return BLADERF_ERR_UNEXPECTED;
        }

        status = write_vcocap(dev, base, vcocap, vcocap_reg_state);
        if (status != 0) {
            return status;
        }

        printf("Waiting for VTUNE HIGH @ VCOCAP=%u\r\n", vcocap);
        status = wait_for_vtune_value(dev, base, VCO_HIGH,
                                      &vcocap, vcocap_reg_state);

        if (status == 0) {
            printf("Walking VTUNE HIGH to NORM from VCOCAP=%u,\r\n", vcocap);
            status = vtune_high_to_norm(dev, base, vcocap, vcocap_reg_state,
                                        &vtune_high_limit);
        }
    }

    if (status == 0) {
        vcocap = vtune_high_limit + (vtune_low_limit - vtune_high_limit) / 2;

        printf("VTUNE LOW:   %u\r\n", vtune_low_limit);
        printf("VTUNE NORM:  %u\r\n", vcocap);
        printf("VTUNE Est:   %u (%d)\r\n",
                     vcocap_est, (int) vcocap_est - vcocap);
        printf("VTUNE HIGH:  %u\r\n", vtune_high_limit);

#       if LMS_COUNT_BUSY_WAITS
        printf("Busy waits:  %u\r\n", busy_wait_count);
        printf("Busy us:     %u\r\n", busy_wait_duration);
#       endif

        status = write_vcocap(dev, base, vcocap, vcocap_reg_state);
        if (status != 0) {
            return status;
        }

        /* Inform the caller of what we converged to */
        *vcocap_result = vcocap;

        status = get_vtune(dev, base, VTUNE_DELAY_SMALL, &vtune);
        if (status != 0) {
            return status;
        }

        PRINT_BUSY_WAIT_INFO();

        if (vtune != VCO_NORM) {
           status = BLADERF_ERR_UNEXPECTED;
           printf("Final VCOCAP=%u is not in VTUNE NORM region.\r\n", vcocap);
        }
    }

    return status;
}

static inline int is_loopback_enabled(void *dev)
{
    bladerf_loopback loopback;
    int status;

    status = lms_get_loopback_mode(dev, &loopback);
    if (status != 0) {
        return status;
    }

    return loopback != BLADERF_LB_NONE;
}

int lms_select_pa(void *dev, lms_pa pa)
{
    int status;
    uint8_t data;

    status = LMS_READ(dev, 0x44, &data);

    /* Disable PA1, PA2, and AUX PA - we'll enable as requested below. */
    data &= ~0x1C;

    /* AUX PA powered down */
    data |= (1 << 2);

    switch (pa) {
        case PA_AUX:
            data &= ~(1 << 2);  /* Power up the AUX PA */
            break;

        case PA_1:
            data |= (2 << 2);   /* PA_EN[2:0] = 010 - Enable PA1 */
            break;

        case PA_2:
            data |= (4 << 2);   /* PA_EN[2:0] = 100 - Enable PA2 */
            break;

        case PA_NONE:
            break;

        default:
            //assert(!"Invalid PA selection");
            status = BLADERF_ERR_INVAL;
    }

    if (status == 0) {
        status = LMS_WRITE(dev, 0x44, data);
    }

    return status;

};

/* Select which LNA to enable */
int lms_select_lna(void *dev, lms_lna lna)
{
    int status;
    uint8_t data;

    status = LMS_READ(dev, 0x75, &data);
    if (status != 0) {
        return status;
    }

    data &= ~(3 << 4);
    data |= ((lna & 3) << 4);

    return LMS_WRITE(dev, 0x75, data);
}

int lms_select_band(void *dev, bladerf_module module, bool low_band)
{
    int status;

    /* If loopback mode disabled, avoid changing the PA or LNA selection,
     * as these need to remain are powered down or disabled */
    status = is_loopback_enabled(dev);
    if (status < 0) {
        return status;
    } else if (status > 0) {
        return 0;
    }

    if (module == BLADERF_MODULE_TX)
    {
    	// SYRTEM
    	// lms_pa pa = low_band ? PA_1 : PA_2;
    	lms_pa pa = PA_1;	// force PA1 : High band output (1500 – 3800 MHz)
        status = lms_select_pa(dev, pa);
    } else
    {
        lms_lna lna = low_band ? LNA_1 : LNA_2;
        status = lms_select_lna(dev, lna);
    }

    return status;
}

/* Register 0x08:  RF loopback config and additional BB config
 *
 * LBRFEN[3:0] @ [3:0]
 *  0000 - RF loopback disabled
 *  0001 - TXMIX output connected to LNA1 path
 *  0010 - TXMIX output connected to LNA2 path
 *  0011 - TXMIX output connected to LNA3 path
 *  else - Reserved
 *
 * LBEN_OPIN @ [4]
 *  0   - Disabled
 *  1   - TX BB loopback signal is connected to RX output pins
 *
 * LBEN_VGA2IN @ [5]
 *  0   - Disabled
 *  1   - TX BB loopback signal is connected to RXVGA2 input
 *
 * LBEN_LPFIN @ [6]
 *  0   - Disabled
 *  1   - TX BB loopback signal is connected to RXLPF input
 *
 */
#define LBEN_OPIN   (1 << 4)
#define LBEN_VGA2IN (1 << 5)
#define LBEN_LPFIN  (1 << 6)
#define LBEN_MASK   (LBEN_OPIN | LBEN_VGA2IN | LBEN_LPFIN)

#define LBRFEN_LNA1 1
#define LBRFEN_LNA2 2
#define LBRFEN_LNA3 3
#define LBRFEN_MASK 0xf     /* [3:2] are marked reserved */


/* Register 0x46: Baseband loopback config
 *
 * LOOPBBEN[1:0] @ [3:2]
 *  00 - All Baseband loops opened (default)
 *  01 - TX loopback path connected from TXLPF output
 *  10 - TX loopback path connected from TXVGA1 output
 *  11 - TX loopback path connected from Env/peak detect output
 */
#define LOOPBBEN_TXLPF  (1 << 2)
#define LOOPBBEN_TXVGA  (2 << 2)
#define LOOPBBEN_ENVPK  (3 << 2)
#define LOOBBBEN_MASK   (3 << 2)

int lms_get_loopback_mode(void *dev, bladerf_loopback *loopback)
{
    int status;
    uint8_t lben_lbrfen, loopbben;


    status = LMS_READ(dev, 0x08, &lben_lbrfen);
    if (status != 0) {
        return status;
    }

    status = LMS_READ(dev, 0x46, &loopbben);
    if (status != 0) {
        return status;
    }

    switch (lben_lbrfen & 0x7) {
        case LBRFEN_LNA1:
            *loopback = BLADERF_LB_RF_LNA1;
            return 0;

        case LBRFEN_LNA2:
            *loopback = BLADERF_LB_RF_LNA2;
            return 0;

        case LBRFEN_LNA3:
            *loopback = BLADERF_LB_RF_LNA3;
            return 0;

        default:
            break;
    }

    switch (lben_lbrfen & LBEN_MASK) {
        case LBEN_VGA2IN:
            if (loopbben & LOOPBBEN_TXLPF) {
                *loopback = BLADERF_LB_BB_TXLPF_RXVGA2;
                return 0;
            } else if (loopbben & LOOPBBEN_TXVGA) {
                *loopback = BLADERF_LB_BB_TXVGA1_RXVGA2;
                return 0;
            }
            break;

        case LBEN_LPFIN:
            if (loopbben & LOOPBBEN_TXLPF) {
                *loopback = BLADERF_LB_BB_TXLPF_RXLPF;
                return 0;
            } else if (loopbben & LOOPBBEN_TXVGA) {
                *loopback = BLADERF_LB_BB_TXVGA1_RXLPF;
                return 0;
            }
            break;

        default:
            break;
    }

    *loopback = BLADERF_LB_NONE;
    return 0;
}

static int write_pll_config(void *dev, bladerf_module module,
                            uint8_t freqsel, bool low_band)
{
    int status;
    uint8_t regval;
    uint8_t selout;
    uint8_t addr;

    if (module == BLADERF_MODULE_TX) {
        addr = 0x15;
    } else {
        addr = 0x25;
    }

    status = LMS_READ(dev, addr, &regval);
    if (status != 0) {
        return status;
    }

    status = is_loopback_enabled(dev);
    if (status < 0) {
        return status;
    }

    if (status == 0) {
        /* Loopback not enabled - update the PLL output buffer. */
        selout = low_band ? 1 : 2;
        regval = (freqsel << 2) | selout;
    } else {
        /* Loopback is enabled - don't touch PLL output buffer. */
        regval = (regval & ~0xfc) | (freqsel << 2);
    }

    return LMS_WRITE(dev, addr, regval);
}

int lms_set_precalculated_frequency(void *dev, bladerf_module mod,
                                    struct lms_freq *f)
{
    /* Select the base address based on which PLL we are configuring */
    const uint8_t base = (mod == BLADERF_MODULE_RX) ? 0x20 : 0x10;

    uint8_t data;
    uint8_t vcocap_reg_state;
    int status, dsm_status;

    /* Utilize atomic writes to the PLL registers, if possible. This
     * "multiwrite" is indicated by the MSB being set. */
    const uint8_t pll_base = base;

    f->vcocap_result = 0xff;

    /* Turn on the DSMs */
    status = LMS_READ(dev, 0x09, &data);
    if (status == 0) {
        data |= 0x05;
        status = LMS_WRITE(dev, 0x09, data);
    }

    if (status != 0) {
        printf("Failed to turn on DSMs\r\n");
        return status;
    }

    /* Write the initial vcocap estimate first to allow for adequate time for
     * VTUNE to stabilize. We need to be sure to keep the upper bits of
     * this register and perform a RMW, as bit 7 is VOVCOREG[0]. */
    status = LMS_READ(dev, base + 9, &vcocap_reg_state);
    if (status != 0) {
        goto error;
    }

    vcocap_reg_state &= ~(0x3f);

    status = write_vcocap(dev, base, f->vcocap, vcocap_reg_state);
    if (status != 0) {
        goto error;
    }

    status = write_pll_config(dev, mod, f->freqsel,
                              (f->flags & LMS_FREQ_FLAGS_LOW_BAND) != 0);
    if (status != 0) {
        goto error;
    }

    data = f->nint >> 1;
    status = LMS_WRITE(dev, pll_base + 0, data);
    if (status != 0) {
        goto error;
    }
    printf("NINT=0x%.2x\r\n", data);

    data = ((f->nint & 1) << 7) | ((f->nfrac >> 16) & 0x7f);
    status = LMS_WRITE(dev, pll_base + 1, data);
    if (status != 0) {
        goto error;
    }
    printf("NFR1=0x%.2x\r\n", data);

    data = ((f->nfrac >> 8) & 0xff);
    status = LMS_WRITE(dev, pll_base + 2, data);
    if (status != 0) {
        goto error;
    }
    printf("NFR2=0x%.2x\r\n", data);

    data = (f->nfrac & 0xff);
    status = LMS_WRITE(dev, pll_base + 3, data);
    if (status != 0) {
        goto error;
    }
    printf("NFR3=0x%.2x\r\n", data);

    /* Perform tuning algorithm unless we've been instructed to just use
     * the VCOCAP hint as-is. */
    if (f->flags & LMS_FREQ_FLAGS_FORCE_VCOCAP) {
        f->vcocap_result = f->vcocap;
    } else {
        /* Walk down VCOCAP values find an optimal values */
        status = tune_vcocap(dev, f->vcocap, base, vcocap_reg_state,
                             &f->vcocap_result);
    }

error:
    /* Turn off the DSMs */
    dsm_status = LMS_READ(dev, 0x09, &data);
    if (dsm_status == 0) {
        data &= ~(0x05);
        dsm_status = LMS_WRITE(dev, 0x09, data);
    }

    return (status == 0) ? dsm_status : status;
}

int band_select(void *dev, bladerf_module module, bool low_band)
{
    int status;
    uint32_t gpio;
    const uint32_t band = low_band ? 2 : 1;

    printf("Selecting %s band.\r\n", low_band ? "low" : "high");

    status = lms_select_band(dev, module, low_band);
    if (status != 0) {
        return status;
    }

    return status;
}

int tuning_set_freq(void *dev, bladerf_module module,
                    unsigned int frequency)
{
	int status;

	const struct dc_cal_tbl *dc_cal = NULL;

	status = lms_set_frequency(dev, module, frequency);
	if (status != 0)
	{
		return status;
	}
	status = band_select(dev, module, frequency < BLADERF_BAND_HIGH);

	return status;
}

int init_device(void *dev)
{
    int			status	= 0;
    uint32_t	val		= 0;

	/* Disable the front ends */
	status = lms_enable_rffe(dev, BLADERF_MODULE_TX, false);
	if (status != 0)
	{
		return status;
	}
	status = lms_enable_rffe(dev, BLADERF_MODULE_RX, false);
	if (status != 0)
	{
		return status;
	}

	/* Set the internal LMS register to enable RX and TX */
	status = LMS_WRITE(dev, 0x05, 0x3e);
	if (status != 0)
	{
		return status;
	}
	/* LMS FAQ FAQ5.27: Improve TX spurious emission performance */
	status = LMS_WRITE(dev, 0x47, 0x40);
	if (status != 0)
	{
		return status;
	}
	/* LMS FAQ FAQ5.27: Improve ADC performance */
	status = LMS_WRITE(dev, 0x59, 0x29);
	if (status != 0)
	{
		return status;
	}
	/* LMS FAQ FAQ5.27: Common mode voltage for ADC */
	status = LMS_WRITE(dev, 0x64, 0x36);
	if (status != 0)
	{
		return status;
	}
	/* LMS FAQ FAQ5.27: Higher LNA Gain */
	status = LMS_WRITE(dev, 0x79, 0x37);
	if (status != 0)
	{
		return status;
	}

	/* Power down DC calibration comparators until they are need, as they
	* have been shown to introduce undesirable artifacts into our signals.
	* (This is documented in the LMS6 FAQ). */
	/* Power down DC offset comparators in DC offset cancellation block.
	 * Should be powered up only when DC offset cancellation algorithm is running. */
	status = lms_set(dev, 0x3f, 0x80);  /* TX LPF DC cal comparator */
	if (status != 0)
	{
		return status;
	}
	/* D_DCOCMP_LP: Power down DC offset comparators in the DC offset cancellation block.
	 * Should be powered up only when DC offset cancellation algorithm is running. */
	status = lms_set(dev, 0x5f, 0x80);  /* RX LPF DC cal comparator FAQ5.26 */
	if (status != 0)
	{
		return status;
	}
	/* PD[7] - DC calibration comparator for VGA2B
	 *    1 – powered down
	 * PD[6] - DC calibration comparator for VGA2A
	 *    1 – powered down */
	status = lms_set(dev, 0x6e, 0xc0);  /* RXVGA2A/B DC cal comparators FAQ5.26 */
	if (status != 0)
	{
		return status;
	}

	/* Configure charge pump current offsets */
	status = lms_config_charge_pumps(dev, BLADERF_MODULE_TX);
	if (status != 0)
	{
		return status;
	}
	status = lms_config_charge_pumps(dev, BLADERF_MODULE_RX);
	if (status != 0)
	{
		return status;
	}

	status = tuning_set_freq(dev, BLADERF_MODULE_TX, 2542000000U);
	if (status != 0)
	{
		return status;
	}

	status = tuning_set_freq(dev, BLADERF_MODULE_RX, 2662000000U);
	if (status != 0)
	{
		return status;
	}

	return status;
}

/** Minimum RXVGA1 gain, in dB */
#define BLADERF_RXVGA1_GAIN_MIN     5

/** Maximum RXVGA1 gain, in dB */
#define BLADERF_RXVGA1_GAIN_MAX     30

/** Minimum RXVGA2 gain, in dB */
#define BLADERF_RXVGA2_GAIN_MIN     0

/** Maximum RXVGA2 gain, in dB */
#define BLADERF_RXVGA2_GAIN_MAX     30

/** Minimum TXVGA1 gain, in dB */
#define BLADERF_TXVGA1_GAIN_MIN     (-35)

/** Maximum TXVGA1 gain, in dB */
#define BLADERF_TXVGA1_GAIN_MAX     (-4)

/** Minimum TXVGA2 gain, in dB */
#define BLADERF_TXVGA2_GAIN_MIN     0

/** Maximum TXVGA2 gain, in dB */
#define BLADERF_TXVGA2_GAIN_MAX     25

/** Minimum sample rate, in Hz */
#define BLADERF_SAMPLERATE_MIN      80000u

/** Maximum recommended sample rate, in Hz */
#define BLADERF_SAMPLERATE_REC_MAX  40000000u

/** Minimum bandwidth, in Hz */
#define BLADERF_BANDWIDTH_MIN       1500000u

/** Maximum bandwidth, in Hz */
#define BLADERF_BANDWIDTH_MAX       28000000u

int lms_txvga2_set_gain(void *dev, int gain_int)
{
    int status;
    uint8_t data;
    int8_t gain;

    if (gain_int > BLADERF_TXVGA2_GAIN_MAX) {
        gain = BLADERF_TXVGA2_GAIN_MAX;
        printf("Clamping TXVGA2 gain to %ddB\r\n", gain);
    } else if (gain_int < BLADERF_TXVGA2_GAIN_MIN) {
        gain = 0;
        printf("Clamping TXVGA2 gain to %ddB\r\n", gain);
    } else {
        gain = gain_int;
    }

    status = LMS_READ(dev, 0x45, &data);
    if (status == 0) {
        data &= ~(0x1f << 3);
        data |= ((gain & 0x1f) << 3);
        status = LMS_WRITE(dev, 0x45, data);
    }

    return status;
}

int lms_rxvga2_enable(void *dev, bool enable)
{
    int status;
    uint8_t data;

    status = LMS_READ(dev, 0x64, &data);
    if (status != 0) {
        return status;
    }

    if (enable) {
        data |= (1 << 1);
    } else {
        data &= ~(1 << 1);
    }

    return LMS_WRITE(dev, 0x64, data);
}

int lms_rxvga2_set_gain(void *dev, int gain)
{
    if (gain > BLADERF_RXVGA2_GAIN_MAX) {
        gain = BLADERF_RXVGA2_GAIN_MAX;
        printf("Clamping RXVGA2 gain to %ddB\r\n", gain);
    } else if (gain < BLADERF_RXVGA2_GAIN_MIN) {
        gain = BLADERF_RXVGA2_GAIN_MIN;
        printf("Clamping RXVGA2 gain to %ddB\r\n", gain);
    }

    /* 3 dB per register code */
    return LMS_WRITE(dev, 0x65, gain / 3);
}

struct dc_cal_state {
    uint8_t clk_en;                 /* Backup of clock enables */

    uint8_t reg0x72;                /* Register backup */

    bladerf_lna_gain lna_gain;      /* Backup of gain values */
    int rxvga1_gain;
    int rxvga2_gain;

    uint8_t base_addr;              /* Base address of DC cal regs */
    unsigned int num_submodules;    /* # of DC cal submodules to operate on */

    int rxvga1_curr_gain;           /* Current gains used in retry loops */
    int rxvga2_curr_gain;
};

int lms_lna_get_gain(void *dev, bladerf_lna_gain *gain)
{
    int status;
    uint8_t data;

    status = LMS_READ(dev, 0x75, &data);
    if (status == 0) {
        data >>= 6;
        data &= 3;
        *gain = (bladerf_lna_gain)data;

        if (*gain == BLADERF_LNA_GAIN_UNKNOWN) {
            status = BLADERF_ERR_INVAL;
        }
    }

    return status;
}

/*
 * The LMS FAQ (Rev 1.0r10, Section 5.20) states that the RXVGA1 codes may be
 * converted to dB via:
 *      value_db = 20 * log10(127 / (127 - code))
 *
 * However, an offset of 5 appears to be required, yielding:
 *      value_db =  5 + 20 * log10(127 / (127 - code))
 *
 */
static const uint8_t rxvga1_lut_code2val[] = {
    5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
    6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,
    8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10,
    10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13,
    13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 17,
    17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 21, 21, 22, 22, 22, 23, 24, 24,
    25, 25, 26, 27, 28, 29, 30
};

/* The closest values from the above forumla have been selected.
 * indicides 0 - 4 are clamped to 5dB */
static const uint8_t rxvga1_lut_val2code[] = {
    2,  2,  2,  2,   2,   2,   14,  26,  37,  47,  56,  63,  70,  76,  82,  87,
    91, 95, 99, 102, 104, 107, 109, 111, 113, 114, 116, 117, 118, 119, 120,
};

int lms_rxvga1_get_gain(void *dev, int *gain)
{
    uint8_t data;
    int status = LMS_READ(dev, 0x76, &data);

    if (status == 0) {
        data &= 0x7f;
        if (data > 120) {
            data = 120;
        }

        *gain = rxvga1_lut_code2val[data];
    }

    return status;
}

int lms_rxvga2_get_gain(void *dev, int *gain)
{

    uint8_t data;
    const int status = LMS_READ(dev, 0x65, &data);

    if (status == 0) {
        /* 3 dB per code */
        data *= 3;
        *gain = data;
    }

    return status;
}

int lms_lna_set_gain(void *dev, bladerf_lna_gain gain)
{
    int status;
    uint8_t data;

    if (gain == BLADERF_LNA_GAIN_BYPASS || gain == BLADERF_LNA_GAIN_MID ||
        gain == BLADERF_LNA_GAIN_MAX) {

        status = LMS_READ(dev, 0x75, &data);
        if (status == 0) {
            data &= ~(3 << 6);          /* Clear out previous gain setting */
            data |= ((gain & 3) << 6);  /* Update gain value */
            status = LMS_WRITE(dev, 0x75, data);
        }

    } else {
        status = BLADERF_ERR_INVAL;
    }

    return status;
}

int lms_rxvga1_set_gain(void *dev, int gain)
{
    if (gain > BLADERF_RXVGA1_GAIN_MAX) {
        gain = BLADERF_RXVGA1_GAIN_MAX;
        printf("Clamping RXVGA1 gain to %ddB\r\n", gain);
    } else if (gain < BLADERF_RXVGA1_GAIN_MIN) {
        gain = BLADERF_RXVGA1_GAIN_MIN;
        printf("Clamping RXVGA1 gain to %ddB\r\n", gain);
    }

    return LMS_WRITE(dev, 0x76, rxvga1_lut_val2code[gain]);
}

static inline int dc_cal_backup(void *dev,
                                bladerf_cal_module module,
                                struct dc_cal_state *state)
{
    int status;

    memset(state, 0, sizeof(state[0]));

    status = LMS_READ(dev, 0x09, &state->clk_en);
    if (status != 0) {
        return status;
    }

    if (module == BLADERF_DC_CAL_RX_LPF || module == BLADERF_DC_CAL_RXVGA2) {
        status = LMS_READ(dev, 0x72, &state->reg0x72);
        if (status != 0) {
            return status;
        }

        status = lms_lna_get_gain(dev, &state->lna_gain);
        if (status != 0) {
            return status;
        }

        status = lms_rxvga1_get_gain(dev, &state->rxvga1_gain);
        if (status != 0) {
            return status;
        }

        status = lms_rxvga2_get_gain(dev, &state->rxvga2_gain);
        if (status != 0) {
            return status;
        }
    }

    return 0;
}

static int dc_cal_module_init(void *dev,
                                     bladerf_cal_module module,
                                     struct dc_cal_state *state)
{
    int status;
    uint8_t cal_clock;
    uint8_t val;

    switch (module) {
        case BLADERF_DC_CAL_LPF_TUNING:
            cal_clock = (1 << 5);  /* CLK_EN[5] - LPF CAL Clock */
            state->base_addr = 0x00;
            state->num_submodules = 1;
            break;

        case BLADERF_DC_CAL_TX_LPF:
            cal_clock = (1 << 1);  /* CLK_EN[1] - TX LPF DCCAL Clock */
            state->base_addr = 0x30;
            state->num_submodules = 2;
            break;

        case BLADERF_DC_CAL_RX_LPF:
            cal_clock = (1 << 3);  /* CLK_EN[3] - RX LPF DCCAL Clock */
            state->base_addr = 0x50;
            state->num_submodules = 2;
            break;

        case BLADERF_DC_CAL_RXVGA2:
            cal_clock = (1 << 4);  /* CLK_EN[4] - RX VGA2 DCCAL Clock */
            state->base_addr = 0x60;
            state->num_submodules = 5;
            break;

        default:
            return BLADERF_ERR_INVAL;
    }

    /* Enable the appropriate clock based on the module */
    status = LMS_WRITE(dev, 0x09, state->clk_en | cal_clock);
    if (status != 0) {
        return status;
    }

    switch (module) {

        case BLADERF_DC_CAL_LPF_TUNING:
            /* Nothing special to do */
            break;

        case BLADERF_DC_CAL_RX_LPF:
        case BLADERF_DC_CAL_RXVGA2:
            /* FAQ 5.26 (rev 1.0r10) notes that the DC comparators should be
             * powered up when performing DC calibration, and then powered down
             * afterwards to improve receiver linearity */
            if (module == BLADERF_DC_CAL_RXVGA2) {
                status = lms_clear(dev, 0x6e, (3 << 6));
                if (status != 0) {
                    return status;
                }
            } else {
                /* Power up RX LPF DC calibration comparator */
                status = lms_clear(dev, 0x5f, (1 << 7));
                if (status != 0) {
                    return status;
                }
            }

            /* Disconnect LNA from the RXMIX input by opening up the
             * INLOAD_LNA_RXFE switch. This should help reduce external
             * interference while calibrating */
            val = state->reg0x72 & ~(1 << 7);
            status = LMS_WRITE(dev, 0x72, val);
            if (status != 0) {
                return status;
            }

            /* Attempt to calibrate at max gain. */
            status = lms_lna_set_gain(dev, BLADERF_LNA_GAIN_MAX);
            if (status != 0) {
                return status;
            }

            state->rxvga1_curr_gain = BLADERF_RXVGA1_GAIN_MAX;
            status = lms_rxvga1_set_gain(dev, state->rxvga1_curr_gain);
            if (status != 0) {
                return status;
            }

            state->rxvga2_curr_gain = BLADERF_RXVGA2_GAIN_MAX;
            status = lms_rxvga2_set_gain(dev, state->rxvga2_curr_gain);
            if (status != 0) {
                return status;
            }

            break;


        case BLADERF_DC_CAL_TX_LPF:
            /* FAQ item 4.1 notes that the DAC should be turned off or set
             * to generate minimum DC */
            status = lms_set(dev, 0x36, (1 << 7));
            if (status != 0) {
                return status;
            }

            /* Ensure TX LPF DC calibration comparator is powered up */
            status = lms_clear(dev, 0x3f, (1 << 7));
            if (status != 0) {
                return status;
            }
            break;

        default:
            //assert(!"Invalid module");
            status = BLADERF_ERR_INVAL;
    }

    return status;
}

static int lms_dc_cal_loop(void *dev, uint8_t base,
                           uint8_t cal_address, uint8_t dc_cntval,
                           uint8_t *dc_regval)
{
    int status;
    uint8_t i, val;
    bool done = false;
    const unsigned int max_cal_count = 25;

    printf("Calibrating module %2.2x:%2.2x\r\n", base, cal_address);

    /* Set the calibration address for the block, and start it up */
    status = LMS_READ(dev, base + 0x03, &val);
    if (status != 0) {
        return status;
    }

    val &= ~(0x07);
    val |= cal_address&0x07;

    status = LMS_WRITE(dev, base + 0x03, val);
    if (status != 0) {
        return status;
    }

    /* Set and latch the DC_CNTVAL  */
    status = LMS_WRITE(dev, base + 0x02, dc_cntval);
    if (status != 0) {
        return status;
    }

    val |= (1 << 4);
    status = LMS_WRITE(dev, base + 0x03, val);
    if (status != 0) {
        return status;
    }

    val &= ~(1 << 4);
    status = LMS_WRITE(dev, base + 0x03, val);
    if (status != 0) {
        return status;
    }


    /* Start the calibration by toggling DC_START_CLBR */
    val |= (1 << 5);
    status = LMS_WRITE(dev, base + 0x03, val);
    if (status != 0) {
        return status;
    }

    val &= ~(1 << 5);
    status = LMS_WRITE(dev, base + 0x03, val);
    if (status != 0) {
        return status;
    }

    /* Main loop checking the calibration */
    for (i = 0 ; i < max_cal_count && !done; i++) {
        /* Read active low DC_CLBR_DONE */
        status = LMS_READ(dev, base + 0x01, &val);
        if (status != 0) {
            return status;
        }

        /* Check if calibration is done */
        if (((val >> 1) & 1) == 0) {
            done = true;
            /* Per LMS FAQ item 4.7, we should check DC_REG_VAL, as
             * DC_LOCK is not a reliable indicator */
            status = LMS_READ(dev, base, dc_regval);
            if (status == 0) {
                *dc_regval &= 0x3f;
            }
        }
    }

    if (done == false) {
        printf("DC calibration loop did not converge.\r\n");
        status = BLADERF_ERR_UNEXPECTED;
    } else {
    	printf( "DC_REGVAL: %d\r\n", *dc_regval );
    }

    return status;
}

static int dc_cal_submodule(void *dev,
                                   bladerf_cal_module module,
                                   unsigned int submodule,
                                   struct dc_cal_state *state,
                                   bool *converged)
{
    int status;
    uint8_t val, dc_regval;

    *converged = false;

    if (module == BLADERF_DC_CAL_RXVGA2) {
        switch (submodule) {
            case 0:
                /* Reset VGA2GAINA and VGA2GAINB to the default power-on values,
                 * in case we're retrying this calibration due to one of the
                 * later submodules failing. For the same reason, RXVGA2 decode
                 * is disabled; it is not used for the RC reference module (0)
                 */

                /* Disable RXVGA2 DECODE */
                status = lms_clear(dev, 0x64, (1 << 0));
                if (status != 0) {
                    return status;
                }

                /* VGA2GAINA = 0, VGA2GAINB = 0 */
                status = LMS_WRITE(dev, 0x68, 0x01);
                if (status != 0) {
                    return status;
                }
                break;

            case 1:
                /* Setup for Stage 1 I and Q channels (submodules 1 and 2) */

                /* Set to direct control signals: RXVGA2 Decode = 1 */
                status = lms_set(dev, 0x64, (1 << 0));
                if (status != 0) {
                    return status;
                }

                /* VGA2GAINA = 0110, VGA2GAINB = 0 */
                val = 0x06;
                status = LMS_WRITE(dev, 0x68, val);
                if (status != 0) {
                    return status;
                }
                break;

            case 2:
                /* No additional changes needed - covered by previous execution
                 * of submodule == 1. */
                break;

            case 3:
                /* Setup for Stage 2 I and Q channels (submodules 3 and 4) */

                /* VGA2GAINA = 0, VGA2GAINB = 0110 */
                val = 0x60;
                status = LMS_WRITE(dev, 0x68, val);
                if (status != 0) {
                    return status;
                }
                break;

            case 4:
                /* No additional changes needed - covered by execution
                 * of submodule == 3 */
                break;

            default:
                //assert(!"Invalid submodule");
                return BLADERF_ERR_UNEXPECTED;
        }
    }

    status = lms_dc_cal_loop(dev, state->base_addr, submodule, 31, &dc_regval);
    if (status != 0) {
        return status;
    }

    if (dc_regval == 31) {
        printf("DC_REGVAL suboptimal value - retrying DC cal loop.\r\n");

        /* FAQ item 4.7 indcates that can retry with DC_CNTVAL reset */
        status = lms_dc_cal_loop(dev, state->base_addr, submodule, 0, &dc_regval);
        if (status != 0) {
            return status;
        } else if (dc_regval == 0) {
            printf("Bad DC_REGVAL detected. DC cal failed.\r\n");
            return 0;
        }
    }

    if (module == BLADERF_DC_CAL_LPF_TUNING) {
        /* Special case for LPF tuning module where results are
         * written to TX/RX LPF DCCAL */

        /* Set the DC level to RX and TX DCCAL modules */
        status = LMS_READ(dev, 0x35, &val);
        if (status == 0) {
            val &= ~(0x3f);
            val |= dc_regval;
            status = LMS_WRITE(dev, 0x35, val);
        }

        if (status != 0) {
            return status;
        }
		
        status = LMS_READ(dev, 0x55, &val);
        if (status == 0) {
            val &= ~(0x3f);
            val |= dc_regval;
            status = LMS_WRITE(dev, 0x55, val);
        }

        if (status != 0) {
            return status;
        }
    }

    *converged = true;
    return 0;
}

static inline int dc_cal_module(void *dev,
                                bladerf_cal_module module,
                                struct dc_cal_state *state,
                                bool *converged)
{
    unsigned int	i;
    int				status		= 0;
//	unsigned char	reg0x70_val	= 0;

    *converged = true;
#if 0	// SYRTEM
	lime_spi_blk_write(cmdcontext_spi,	0x06, 0x0C);
	lime_spi_blk_write(cmdcontext_spi,	0x09, 0xCD);
	lime_spi_blk_write(cmdcontext_spi,	0x0B, 0x09);

	lime_spi_blk_write(cmdcontext_spi,	0x3F, 0x00);

	lime_spi_blk_write(cmdcontext_spi,	0x40, 0x02);
	lime_spi_blk_write(cmdcontext_spi,	0x44, 0x08);
	lime_spi_blk_write(cmdcontext_spi,	0x4B, 0xE3);

	lime_spi_blk_write(cmdcontext_spi,	0x59, 0x21);
	lime_spi_blk_write(cmdcontext_spi,	0x5A, 0x30);
#endif
#if 1
//	lime_spi_blk_read(cmdcontext_spi,	0x70, &reg0x70_val);
//	printf("reg0x70_val=0x%.2x: \r\n", reg0x70_val);
	lime_spi_blk_write(cmdcontext_spi,	0x70, 0x01);
//	lime_spi_blk_write(cmdcontext_spi,	0x72, 0x98);
#endif
#if 1
	printf("**********************************************************************************************\r\n");
	printf("(for i=0; i<num_submodulles; i++) status = dc_cal_submodule(dev, module, i, state, converged);\r\n");
//	lms_read_registers(cmdcontext_spi);
//	lms_read_cal_registers(cmdcontext_spi);
#endif
    for (i = 0; i < state->num_submodules && *converged && status == 0; i++) {
        status = dc_cal_submodule(dev, module, i, state, converged);
    }

    return status;
}

static int dc_cal_retry_adjustment(void *dev,
                                          bladerf_cal_module module,
                                          struct dc_cal_state *state,
                                          bool *limit_reached)
{
    int status = 0;

    switch (module) {
        case BLADERF_DC_CAL_LPF_TUNING:
        case BLADERF_DC_CAL_TX_LPF:
            /* Nothing to adjust here */
            *limit_reached = true;
            break;

        case BLADERF_DC_CAL_RX_LPF:
            if (state->rxvga1_curr_gain > BLADERF_RXVGA1_GAIN_MIN) {
                state->rxvga1_curr_gain -= 1;
                printf("Retrying DC cal with RXVGA1=%d\r\n",
                          state->rxvga1_curr_gain);
                status = lms_rxvga1_set_gain(dev, state->rxvga1_curr_gain);
            } else {
                *limit_reached = true;
            }
            break;

        case BLADERF_DC_CAL_RXVGA2:
            if (state->rxvga1_curr_gain > BLADERF_RXVGA1_GAIN_MIN) {
                state->rxvga1_curr_gain -= 1;
                printf("Retrying DC cal with RXVGA1=%d\r\n",
                          state->rxvga1_curr_gain);
                status = lms_rxvga1_set_gain(dev, state->rxvga1_curr_gain);
            } else if (state->rxvga2_curr_gain > BLADERF_RXVGA2_GAIN_MIN) {
                state->rxvga2_curr_gain -= 3;
                printf("Retrying DC cal with RXVGA2=%d\r\n",
                          state->rxvga2_curr_gain);
                status = lms_rxvga2_set_gain(dev, state->rxvga2_curr_gain);
            } else {
                *limit_reached = true;
            }
            break;

        default:
            *limit_reached = true;
            //assert(!"Invalid module");
            status = BLADERF_ERR_UNEXPECTED;
    }

    if (*limit_reached) {
        printf("DC Cal retry limit reached\r\n");
    }
    return status;
}

static int dc_cal_module_deinit(void *dev,
                                       bladerf_cal_module module,
                                       struct dc_cal_state *state)
{
    int status = 0;

    switch (module) {
        case BLADERF_DC_CAL_LPF_TUNING:
            /* Nothing special to do here */
            break;

        case BLADERF_DC_CAL_RX_LPF:
            /* Power down RX LPF calibration comparator */
            status = lms_set(dev, 0x5f, (1 << 7));
            if (status != 0) {
                return status;
            }
            break;

        case BLADERF_DC_CAL_RXVGA2:
            /* Restore defaults: VGA2GAINA = 1, VGA2GAINB = 0 */
            status = LMS_WRITE(dev, 0x68, 0x01);
            if (status != 0) {
                return status;
            }

            /* Disable decode control signals: RXVGA2 Decode = 0 */
            status = lms_clear(dev, 0x64, (1 << 0));
            if (status != 0) {
                return status;
            }

            /* Power DC comparitors down, per FAQ 5.26 (rev 1.0r10) */
            status = lms_set(dev, 0x6e, (3 << 6));
            if (status != 0) {
                return status;
            }
            break;

        case BLADERF_DC_CAL_TX_LPF:
            /* Power down TX LPF DC calibration comparator */
            status = lms_set(dev, 0x3f, (1 << 7));
            if (status != 0) {
                return status;
            }

            /* Re-enable the DACs */
            status = lms_clear(dev, 0x36, (1 << 7));
            if (status != 0) {
                return status;
            }
            break;

        default:
            //assert(!"Invalid module");
            status = BLADERF_ERR_INVAL;
    }

    return status;
}

static inline int dc_cal_restore(void *dev,
                                 bladerf_cal_module module,
                                 struct dc_cal_state *state)
{
    int status, ret;
    ret = 0;

    status = LMS_WRITE(dev, 0x09, state->clk_en);
    if (status != 0) {
        ret = status;
    }

    if (module == BLADERF_DC_CAL_RX_LPF || module == BLADERF_DC_CAL_RXVGA2) {
        status = LMS_WRITE(dev, 0x72, state->reg0x72);
        if (status != 0 && ret == 0) {
            ret = status;
        }

        status = lms_lna_set_gain(dev, state->lna_gain);
        if (status != 0 && ret == 0) {
            ret = status;
        }

        status = lms_rxvga1_set_gain(dev, state->rxvga1_gain);
        if (status != 0 && ret == 0) {
            ret = status;
        }

        status = lms_rxvga2_set_gain(dev, state->rxvga2_gain);
        if (status != 0) {
            ret = status;
        }
    }

    return ret;
}

int lms_calibrate_dc(void *dev, bladerf_cal_module module)
{
    int status, tmp_status;
    struct dc_cal_state state;
    bool converged, limit_reached;

    status = dc_cal_backup(dev, module, &state);
    if (status != 0) {
        return status;
    }

    status = dc_cal_module_init(dev, module, &state);
    if (status != 0) {
        goto error;
    }

    converged = false;
    limit_reached = false;

    while (!converged && !limit_reached && status == 0) {
        status = dc_cal_module(dev, module, &state, &converged);

        if (status == 0 && !converged) {
            status = dc_cal_retry_adjustment(dev, module, &state,
                                             &limit_reached);
        }
    }

    if (!converged && status == 0) {
        printf("DC Calibration (module=%d) failed to converge.\r\n", module);
        status = BLADERF_ERR_UNEXPECTED;
    }

error:
    tmp_status = dc_cal_module_deinit(dev, module, &state);
    status = (status != 0) ? status : tmp_status;

    tmp_status = dc_cal_restore(dev, module, &state);
    status = (status != 0) ? status : tmp_status;

    return status;
}

/*------------------------------------------------------------------------------
 * DC Calibration routines
 *----------------------------------------------------------------------------*/
int bladerf_calibrate_dc(void *dev, bladerf_cal_module module)
{
    int status;
    status = lms_calibrate_dc(dev, module);
    return status;
}

int bladerf_enable_module(void *dev,
                            bladerf_module m, bool enable)
{
    int status	= 0;

	printf("bladerf_enable_module\r\n");
    if (m != BLADERF_MODULE_RX && m != BLADERF_MODULE_TX) {
        return BLADERF_ERR_INVAL;
    }

    printf("Enable Module: %s - %s\r\n",
                (m == BLADERF_MODULE_RX) ? "RX" : "TX",
                enable ? "True" : "False") ;

//    if (enable == false) {
//        perform_format_deconfig(dev, m);
//    }

    status	= lms_enable_rffe(dev, m, enable);

    return status;
}

int lms_get_frequency(void *dev, bladerf_module mod,
                      struct lms_freq *f)
{
    const uint8_t base = (mod == BLADERF_MODULE_RX) ? 0x20 : 0x10;
    int status;
    uint8_t data;

    status = LMS_READ(dev, base + 0, &data);
    if (status != 0) {
        return status;
    }

    f->nint = ((uint16_t)data) << 1;

    status = LMS_READ(dev, base + 1, &data);
    if (status != 0) {
        return status;
    }

    f->nint |= (data & 0x80) >> 7;
    f->nfrac = ((uint32_t)data & 0x7f) << 16;

    status = LMS_READ(dev, base + 2, &data);
    if (status != 0) {
        return status;
    }

    f->nfrac |= ((uint32_t)data)<<8;

    status = LMS_READ(dev, base + 3, &data);
    if (status != 0) {
        return status;
    }

    f->nfrac |= data;

    status = LMS_READ(dev, base + 5, &data);
    if (status != 0) {
        return status;
    }

    f->freqsel = (data>>2);
    f->x = 1 << ((f->freqsel & 7) - 3);

    status = LMS_READ(dev, base + 9, &data);
    if (status != 0) {
        return status;
    }

    f->vcocap = data & 0x3f;

    return status;
}

static int loopback_tx(void *dev, bladerf_loopback mode)
{
    int status = 0;

    switch(mode) {
        case BLADERF_LB_BB_TXLPF_RXVGA2:
        case BLADERF_LB_BB_TXLPF_RXLPF:
        case BLADERF_LB_BB_TXVGA1_RXVGA2:
        case BLADERF_LB_BB_TXVGA1_RXLPF:
            break;

        case BLADERF_LB_RF_LNA1:
        case BLADERF_LB_RF_LNA2:
        case BLADERF_LB_RF_LNA3:
            status = lms_select_pa(dev, PA_AUX);
            break;

        case BLADERF_LB_NONE:
        {
            struct lms_freq f;

            /* Restore proper settings (PA) for this frequency */
            status = lms_get_frequency(dev, BLADERF_MODULE_TX, &f);
            if (status != 0) {
                return status;
            }

            status = lms_set_frequency(dev, BLADERF_MODULE_TX,
                                       lms_frequency_to_hz(&f));
            if (status != 0) {
                return status;
            }

            status = lms_select_band(dev, BLADERF_MODULE_TX,
                                     lms_frequency_to_hz(&f) < BLADERF_BAND_HIGH);
            break;
        }

        default:
            //assert(!"Invalid loopback mode encountered");
            status = BLADERF_ERR_INVAL;
    }

    return status;
}

int lms_lpf_get_mode(void *dev, bladerf_module mod,
                     bladerf_lpf_mode *mode)
{
    int status;
    const uint8_t reg = (mod == BLADERF_MODULE_RX) ? 0x54 : 0x34;
    uint8_t data_h, data_l;
    bool lpf_enabled, lpf_bypassed;

    status = LMS_READ(dev, reg, &data_l);
    if (status != 0) {
        return status;
    }

    status = LMS_READ(dev, reg + 1, &data_h);
    if (status != 0) {
        return status;
    }

    lpf_enabled  = (data_l & (1 << 1)) != 0;
    lpf_bypassed = (data_h & (1 << 6)) != 0;

    if (lpf_enabled && !lpf_bypassed) {
        *mode = BLADERF_LPF_NORMAL;
    } else if (!lpf_enabled && lpf_bypassed) {
        *mode = BLADERF_LPF_BYPASSED;
    } else if (!lpf_enabled && !lpf_bypassed) {
        *mode = BLADERF_LPF_DISABLED;
    } else {
        printf("Invalid LPF configuration: 0x%02x, 0x%02x\r\n",
                  data_l, data_h);
        status = BLADERF_ERR_INVAL;
    }

    return status;
}

int lms_lpf_set_mode(void *dev, bladerf_module mod,
                     bladerf_lpf_mode mode)
{
    int status;
    const uint8_t reg = (mod == BLADERF_MODULE_RX) ? 0x54 : 0x34;
    uint8_t data_l, data_h;

    status = LMS_READ(dev, reg, &data_l);
    if (status != 0) {
        return status;
    }

    status = LMS_READ(dev, reg + 1, &data_h);
    if (status != 0) {
        return status;
    }

    switch (mode) {
        case BLADERF_LPF_NORMAL:
            data_l |= (1 << 1);     /* Enable LPF */
            data_h &= ~(1 << 6);    /* Disable LPF bypass */
            break;

        case BLADERF_LPF_BYPASSED:
            data_l &= ~(1 << 1);    /* Power down LPF */
            data_h |= (1 << 6);     /* Enable LPF bypass */
            break;

        case BLADERF_LPF_DISABLED:
            data_l &= ~(1 << 1);    /* Power down LPF */
            data_h &= ~(1 << 6);    /* Disable LPF bypass */
            break;

        default:
            printf("Invalid LPF mode: %d\r\n", mode);
            return BLADERF_ERR_INVAL;
    }

    status = LMS_WRITE(dev, reg, data_l);
    if (status != 0) {
        return status;
    }

    status = LMS_WRITE(dev, reg + 1, data_h);
    return status;
}

int lms_rxvga1_enable(void *dev, bool enable)
{
    int status;
    uint8_t data;

    status = LMS_READ(dev, 0x7d, &data);
    if (status != 0) {
        return status;
    }

    if (enable) {
        data &= ~(1 << 3);
    } else {
        data |= (1 << 3);
    }

    return LMS_WRITE(dev, 0x7d, data);
}

static inline int enable_lna_power(void *dev, bool enable)
{
    int status;
    uint8_t regval;

    /* Magic test register to power down LNAs */
    status = LMS_READ(dev, 0x7d, &regval);
    if (status != 0) {
        return status;
    }

    if (enable) {
        regval &= ~(1 << 0);
    } else {
        regval |= (1 << 0);
    }

    status = LMS_WRITE(dev, 0x7d, regval);
    if (status != 0) {
        return status;
    }

    /* Decode test registers */
    status = LMS_READ(dev, 0x70, &regval);
    if (status != 0) {
        return status;
    }

    if (enable) {
        regval &= ~(1 << 1);
    } else {
        regval |= (1 << 1);
    }

    return LMS_WRITE(dev, 0x70, regval);
}

static inline int enable_rf_loopback_switch(void *dev, bool enable)
{
    int status;
    uint8_t regval;

    status = LMS_READ(dev, 0x0b, &regval);
    if (status != 0) {
        return status;
    }

    if (enable) {
        regval |= (1 << 0);
    } else {
        regval &= ~(1 << 0);
    }

    return LMS_WRITE(dev, 0x0b, regval);
}

static int loopback_rx(void *dev, bladerf_loopback mode)
{
    int status;
    bladerf_lpf_mode lpf_mode;
    uint8_t lna;
    uint8_t regval;

    status = lms_lpf_get_mode(dev, BLADERF_MODULE_RX, &lpf_mode);
    if (status != 0) {
        return status;
    }

    switch (mode) {
        case BLADERF_LB_BB_TXLPF_RXVGA2:
        case BLADERF_LB_BB_TXVGA1_RXVGA2:

            /* Ensure RXVGA2 is enabled */
            status = lms_rxvga2_enable(dev, true);
            if (status != 0) {
                return status;
            }

            /* RXLPF must be disabled */
            status = lms_lpf_set_mode(dev, BLADERF_MODULE_RX,
                                      BLADERF_LPF_DISABLED);
            if (status != 0) {
                return status;
            }
            break;

        case BLADERF_LB_BB_TXLPF_RXLPF:
        case BLADERF_LB_BB_TXVGA1_RXLPF:

            /* RXVGA1 must be disabled */
            status = lms_rxvga1_enable(dev, false);
            if (status != 0) {
                return status;
            }

            /* Enable the RXLPF if needed */
            if (lpf_mode == BLADERF_LPF_DISABLED) {
                status = lms_lpf_set_mode(dev, BLADERF_MODULE_RX,
                        BLADERF_LPF_NORMAL);
                if (status != 0) {
                    return status;
                }
            }

            /* Ensure RXVGA2 is enabled */
            status = lms_rxvga2_enable(dev, true);
            if (status != 0) {
                return status;
            }

            break;

        case BLADERF_LB_RF_LNA1:
        case BLADERF_LB_RF_LNA2:
        case BLADERF_LB_RF_LNA3:
            lna = mode - BLADERF_LB_RF_LNA1 + 1;
            //assert(lna >= 1 && lna <= 3);

            /* Power down LNAs */
            status = enable_lna_power(dev, false);
            if (status != 0) {
                return status;
            }

            /* Ensure RXVGA1 is enabled */
            status = lms_rxvga1_enable(dev, true);
            if (status != 0) {
                return status;
            }

            /* Enable the RXLPF if needed */
            if (lpf_mode == BLADERF_LPF_DISABLED) {
                status = lms_lpf_set_mode(dev, BLADERF_MODULE_RX,
                        BLADERF_LPF_NORMAL);
                if (status != 0) {
                    return status;
                }
            }

            /* Ensure RXVGA2 is enabled */
            status = lms_rxvga2_enable(dev, true);
            if (status != 0) {
                return status;
            }

            /* Select output buffer in RX PLL and select the desired LNA */
            status = LMS_READ(dev, 0x25, &regval);
            if (status != 0) {
                return status;
            }

            regval &= ~0x03;
            regval |= lna;

            status = LMS_WRITE(dev, 0x25, regval);
            if (status != 0) {
                return status;
            }

            status = lms_select_lna(dev, (lms_lna) lna);
            if (status != 0) {
                return status;
            }

            /* Enable RF loopback switch */
            status = enable_rf_loopback_switch(dev, true);
            if (status != 0) {
                return status;
            }

            break;

        case BLADERF_LB_NONE:
        {
            struct lms_freq f;

            /* Ensure all RX blocks are enabled */
            status = lms_rxvga1_enable(dev, true);
            if (status != 0) {
                return status;
            }

            if (lpf_mode == BLADERF_LPF_DISABLED) {
                status = lms_lpf_set_mode(dev, BLADERF_MODULE_RX,
                        BLADERF_LPF_NORMAL);
                if (status != 0) {
                    return status;
                }
            }

            status = lms_rxvga2_enable(dev, true);
            if (status != 0) {
                return status;
            }

            /* Disable RF loopback switch */
            status = enable_rf_loopback_switch(dev, false);
            if (status != 0) {
                return status;
            }

            /* Power up LNAs */
            status = enable_lna_power(dev, true);
            if (status != 0) {
                return status;
            }

            /* Restore proper settings (LNA, RX PLL) for this frequency */
            status = lms_get_frequency(dev, BLADERF_MODULE_RX, &f);
            if (status != 0) {
                return status;
            }

            status = lms_set_frequency(dev, BLADERF_MODULE_RX,
                                       lms_frequency_to_hz(&f));
            if (status != 0) {
                return status;
            }


            status = lms_select_band(dev, BLADERF_MODULE_RX,
                                     lms_frequency_to_hz(&f) < BLADERF_BAND_HIGH);
            break;
        }

        default:
            //assert(!"Invalid loopback mode encountered");
            status = BLADERF_ERR_INVAL;
    }

    return status;
}

static int loopback_path(void *dev, bladerf_loopback mode)
{
    int status;
    uint8_t loopbben, lben_lbrf;

    status = LMS_READ(dev, 0x46, &loopbben);
    if (status != 0) {
        return status;
    }

    status = LMS_READ(dev, 0x08, &lben_lbrf);
    if (status != 0) {
        return status;
    }

    /* Default to baseband loopback being disabled  */
    loopbben &= ~LOOBBBEN_MASK;

    /* Default to RF and BB loopback options being disabled */
    lben_lbrf &= ~(LBRFEN_MASK | LBEN_MASK);

    switch(mode) {
        case BLADERF_LB_BB_TXLPF_RXVGA2:
            loopbben |= LOOPBBEN_TXLPF;
            lben_lbrf |= LBEN_VGA2IN;
            break;

        case BLADERF_LB_BB_TXLPF_RXLPF:
            loopbben |= LOOPBBEN_TXLPF;
            lben_lbrf |= LBEN_LPFIN;
            break;

        case BLADERF_LB_BB_TXVGA1_RXVGA2:
            loopbben |= LOOPBBEN_TXVGA;
            lben_lbrf |= LBEN_VGA2IN;
            break;

        case BLADERF_LB_BB_TXVGA1_RXLPF:
            loopbben |= LOOPBBEN_TXVGA;
            lben_lbrf |= LBEN_LPFIN;
            break;

        case BLADERF_LB_RF_LNA1:
            lben_lbrf |= LBRFEN_LNA1;
            break;

        case BLADERF_LB_RF_LNA2:
            lben_lbrf |= LBRFEN_LNA2;
            break;

        case BLADERF_LB_RF_LNA3:
            lben_lbrf |= LBRFEN_LNA3;
            break;

        case BLADERF_LB_NONE:
            break;

        default:
            return BLADERF_ERR_INVAL;
    }

    status = LMS_WRITE(dev, 0x46, loopbben);
    if (status == 0) {
        status = LMS_WRITE(dev, 0x08, lben_lbrf);
    }

    return status;
}

int lms_set_loopback_mode(void *dev, bladerf_loopback mode)
{
    int status;

    /* Verify a valid mode is provided before shutting anything down */
    switch (mode) {
        case BLADERF_LB_BB_TXLPF_RXVGA2:
        case BLADERF_LB_BB_TXLPF_RXLPF:
        case BLADERF_LB_BB_TXVGA1_RXVGA2:
        case BLADERF_LB_BB_TXVGA1_RXLPF:
        case BLADERF_LB_RF_LNA1:
        case BLADERF_LB_RF_LNA2:
        case BLADERF_LB_RF_LNA3:
        case BLADERF_LB_NONE:
            break;

        default:
            return BLADERF_ERR_INVAL;
    }

    /* Disable all PA/LNAs while entering loopback mode or making changes */
    status = lms_select_pa(dev, PA_NONE);
    if (status != 0) {
        return status;
    }

    status = lms_select_lna(dev, LNA_NONE);
    if (status != 0) {
        return status;
    }

    /* Disconnect loopback paths while we re-configure blocks */
    status = loopback_path(dev, BLADERF_LB_NONE);
    if (status != 0) {
        return status;
    }

    /* Configure the RX side of the loopback path */
    status = loopback_rx(dev, mode);
    if (status != 0) {
        return status;
    }

    /* Configure the TX side of the path */
    status = loopback_tx(dev, mode);
    if (status != 0) {
        return status;
    }

    /* Configure "switches" along the loopback path */
    status = loopback_path(dev, mode);
    if (status != 0) {
        return status;
    }

    return 0;
}

int bladerf_set_loopback(void *dev, bladerf_loopback l)
{
    int status;

    status =  lms_set_loopback_mode(dev, l);

    return status;
}

int bladerf_get_loopback(void *dev, bladerf_loopback *l)
{
    int status = BLADERF_ERR_UNEXPECTED;
    *l = BLADERF_LB_NONE;

    if (*l == BLADERF_LB_NONE) {
        status = lms_get_loopback_mode(dev, l);
    }

    return status;
}

/* We've found that running samples through the LMS6 tends to be required
 * for the TX LPF calibration to converge */
static inline int tx_lpf_dummy_tx(void *dev)
{
	int								status;
	int								retval			= 0;
	struct bladerf_metadata			meta;
	int16_t							zero_sample[]	= { 0, 0 };
	bladerf_loopback				loopback_backup;
	struct bladerf_rational_rate	sample_rate_backup;

	memset(&meta, 0, sizeof(meta));

	status = bladerf_get_loopback(dev, &loopback_backup);
	if (status != 0)
	{
		return status;
	}

//    status = bladerf_get_rational_sample_rate(dev, BLADERF_MODULE_TX,
//                                              &sample_rate_backup);
//    if (status != 0) {
//        return status;
//    }

	status = bladerf_set_loopback(dev, BLADERF_LB_BB_TXVGA1_RXVGA2);
	if (status != 0)
	{
		goto out;
	}

//    status = bladerf_set_sample_rate(dev, BLADERF_MODULE_TX, 3000000, NULL);
//    if (status != 0) {
//        goto out;
//    }

//    status = bladerf_sync_config(dev, BLADERF_MODULE_TX,
//                                 BLADERF_FORMAT_SC16_Q11_META,
//                                 64, 16384, 16, 1000);
//    if (status != 0) {
//        goto out;
//    }

	status = bladerf_enable_module(dev, BLADERF_MODULE_TX, true);
	if (status != 0)
	{
		goto out;
	}

	meta.flags = BLADERF_META_FLAG_TX_BURST_START |
	BLADERF_META_FLAG_TX_BURST_END   |
	BLADERF_META_FLAG_TX_NOW;

//    status = bladerf_sync_tx(dev, zero_sample, 1, &meta, 2000);
//    if (status != 0) {
//        goto out;
//    }

out:
	status = bladerf_enable_module(dev, BLADERF_MODULE_TX, false);
	if (status != 0 && retval == 0)
	{
		retval = status;
	}

//    status = bladerf_set_rational_sample_rate(dev, BLADERF_MODULE_TX,
//                                              &sample_rate_backup, NULL);
//    if (status != 0 && retval == 0) {
//        retval = status;
//    }

	status = bladerf_set_loopback(dev, loopback_backup);
	if (status != 0 && retval == 0)
	{
		retval = status;
	}

	return retval;
}

exmimo_config_t *p_exmimo_config;
exmimo_id_t     *p_exmimo_id;

unsigned int build_rflocal(txi, txq, rxi, rxq)
{
  return (txi + txq<<6 + rxi<<12 + rxq<<18);
}
unsigned int build_rfdc(int dcoff_i_rxfe, int dcoff_q_rxfe)
{
  return (dcoff_i_rxfe + dcoff_q_rxfe<<8);
}

void test_config(int card, int ant, unsigned int rf_mode)
{
  p_exmimo_config->framing.eNB_flag		= 0;
  p_exmimo_config->framing.tdd_config	= 0;

  p_exmimo_config->rf.rf_freq_rx[ant]	= 2662000000;
  p_exmimo_config->rf.rf_freq_tx[ant]	= 2542000000;
  p_exmimo_config->rf.rx_gain[ant][0]	= 30;
  p_exmimo_config->rf.tx_gain[ant][0]	= 20;
  p_exmimo_config->rf.rf_mode[ant]		= rf_mode;

  p_exmimo_config->rf.rf_local[ant]		= build_rflocal(20,25,26,04);
  p_exmimo_config->rf.rf_rxdc[ant]		= build_rfdc(128, 128);
  p_exmimo_config->rf.rf_vcocal[ant]	= 0xE<<6 + 0xE;

  openair0_dump_config( card );
}

void get_rx_samples_init(int card, unsigned int rf_mode)
{
	int	ant	= 0;

	p_exmimo_config->framing.eNB_flag   		= 0;
	p_exmimo_config->framing.tdd_config 		= DUPLEXMODE_FDD + TXRXSWITCH_LSB;;
	p_exmimo_config->framing.multicard_syncmode = SYNCMODE_FREE;
	for (ant = 0; ant < 4; ant++)
	{
		if (ant)
		{
			p_exmimo_config->rf.rf_freq_rx[ant]				= 0;
			p_exmimo_config->rf.rf_freq_tx[ant]				= 0;
			p_exmimo_config->rf.rf_mode[ant]				= 0;
		}
		else
		{
			p_exmimo_config->rf.rf_freq_rx[ant]				= 2662000000;
			p_exmimo_config->rf.rf_freq_tx[ant]				= 2542000000;
			p_exmimo_config->rf.rf_mode[ant]				= rf_mode;
		}
		p_exmimo_config->framing.resampling_factor[ant] = 2;
		p_exmimo_config->rf.do_autocal[ant]				= 1;
		p_exmimo_config->rf.rx_gain[ant][0]				= (uint32_t) 30;
		p_exmimo_config->rf.tx_gain[ant][0]				= (uint32_t) 20;
		p_exmimo_config->rf.rf_local[ant]				= build_rflocal(20,25,26,04);
		p_exmimo_config->rf.rf_rxdc[ant]				= build_rfdc(128, 128);
		p_exmimo_config->rf.rf_vcocal[ant]				= 0xE<<6 + 0xE;

		p_exmimo_config->rf.rffe_gain_txlow[ant]		= 31;
		p_exmimo_config->rf.rffe_gain_txhigh[ant]		= 31;
		p_exmimo_config->rf.rffe_gain_rxfinal[ant]		= 31;
		p_exmimo_config->rf.rffe_gain_rxlow[ant]		= 63;
		p_exmimo_config->rf.rffe_band_mode[ant]			= TVWS_TDD;
	}

	openair0_dump_config(card);
}

void get_rx_samples(int card)
{
	short	*dump_sig	= NULL;
	int		j			= 0;

//	printf("get_rx_samples start\r\n");
	openair0_get_frame(card);

	dump_sig = (short*) openair0_exmimo_pci[0].adc_head[0];
	printf("dump_sig = 0x%08lx\r\n", (long unsigned int)dump_sig);

	for( j=0; j<32;j++ )	
	{
		printf("i=%d rx_sig[I]=0x%hx, rx_sig[Q]=0x%hx, Ox%lx\r\n", j, dump_sig[j*2], dump_sig[j*2+1], (unsigned long)dump_sig[j*2]);
	}

//	printf("get_rx_samples stop\r\n");
}

int cal_dc_init(void *dev)
{
	int				ret			= 0;
	int				card		= 0;
	int				ant			= 0;
	unsigned int	my_rf_mode	= 0;
	unsigned int	*p_rx_ant0	= NULL;
	unsigned int	*p_tx_ant0	= NULL;

	my_rf_mode =  (RXEN + TXEN + TXLPFNORM + TXLPFEN + TXLPF25 + RXLPFNORM + RXLPFEN + RXLPF25 + LNA2ON + LNAMax + RFBBNORM);
	my_rf_mode += (DMAMODE_RX + DMAMODE_TX);

	printf ("Detected %d number of cards.\r\n", openair0_num_detected_cards);
	card = 0;
	ant  = 0;
	printf ("Will configure card %d, antenna %d\r\n", card, ant);
	p_exmimo_config = openair0_exmimo_pci[card].exmimo_config_ptr;
	p_exmimo_id     = openair0_exmimo_pci[card].exmimo_id_ptr;

	printf("Card %d: ExpressMIMO %d, HW Rev %d, SW Rev 0x%d\r\n", card, p_exmimo_id->board_exmimoversion, p_exmimo_id->board_hwrev, p_exmimo_id->board_swrev);
	//read_firmware_buffer(card);

	// pointer to data
	p_rx_ant0 = openair0_exmimo_pci[card].adc_head[ant];
	p_tx_ant0 = openair0_exmimo_pci[card].dac_head[ant];

//	memset(p_rx_ant0, 0, 76800);

	get_rx_samples_init(card, my_rf_mode);

	return 0;
}

int cal_dc_deinit(void *dev)
{
	openair0_stop_without_reset(0);
	return 0;
}

void read_firmware_buffer(int card)
{
	int i;
	unsigned int *p= (unsigned int *) (openair0_exmimo_pci[card].firmware_block_ptr);
	printf("firmware_buffer: \r\n");

	for (i=0; i<0x30; i++)
		printf("u32 fwbuf[%d]: value=%08X\r\n", i, p[i]);
}

int cal_tx_lpf_init(void *dev)
{
	int				ret;
	int				card;
	int				ant;
	unsigned int	my_rf_mode;
	unsigned int	*p_rx_ant0;
	unsigned int	*p_tx_ant0;

	my_rf_mode =  (RXEN + TXEN + TXLPFNORM + TXLPFEN + TXLPF25 + RXLPFNORM + RXLPFEN + RXLPF25 + LNA2ON + LNAMax + RFBBNORM);
	my_rf_mode +=  DMAMODE_TX;

	printf ("Detected %d number of cards.\r\n", openair0_num_detected_cards);
	card = 0;
	ant  = 0;
	printf ("Will configure card %d, antenna %d\r\n", card, ant);
	p_exmimo_config = openair0_exmimo_pci[card].exmimo_config_ptr;
	p_exmimo_id     = openair0_exmimo_pci[card].exmimo_id_ptr;

	printf("Card %d: ExpressMIMO %d, HW Rev %d, SW Rev 0x%d\r\n", card, p_exmimo_id->board_exmimoversion, p_exmimo_id->board_hwrev, p_exmimo_id->board_swrev);
	//read_firmware_buffer(card);

	// pointer to data
	p_rx_ant0 = openair0_exmimo_pci[ card ].adc_head[ ant ];
	p_tx_ant0 = openair0_exmimo_pci[ card ].dac_head[ ant ];

	memset(p_tx_ant0, 0, 76800);

	test_config(card, ant, my_rf_mode);

	openair0_start_rt_acquisition(card);

	return 0;
}

int cal_tx_lpf_deinit(void *dev)
{
	openair0_stop_without_reset(0);
	return 0;
}

int cal_tx_lpf(void *dev)
{
	int	status;

	status = tx_lpf_dummy_tx(dev);
	if (status == 0)
	{
		status = bladerf_calibrate_dc(dev, BLADERF_DC_CAL_TX_LPF);
	}

	return status;
}

/* from SYRTEM : LIME6002 calibration */
unsigned char	tunevcocap_tx(void *cmdcontext_spi)
{
	unsigned char	retval			= 0xFF;
	unsigned char	reg0x19_val		= 0;
	unsigned char	reg0x1A_val		= 0;
	int				i				= 0;
	unsigned char	vtune			= 0xFF;
	unsigned char	tunevcocap_min	= 0xFF;
	unsigned char	tunevcocap_max	= 0x3F;
	unsigned char	tunevcocap_res	= 0xFF;
	unsigned char	min_is_done		= 0;
	lime_spi_blk_read(cmdcontext_spi,	0x19, &reg0x19_val);
	for (i = 0; i < 64; i++)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x19, (reg0x19_val&0xC0)|i );
		lime_spi_blk_read(cmdcontext_spi,	0x1A, &reg0x1A_val);
		vtune	= (reg0x1A_val>>6)&0x03;
//		printf("i=%d vtune=0x%x reg0x1A_val=0x%.2x\r\n", i, vtune, reg0x1A_val);
		if ( (vtune == 0) && (!min_is_done) )	// pass 0b10 to 0b00
		{
			tunevcocap_min	= i;
			min_is_done		= 1;
		}
		else if (vtune == 1)	// pass 0b00 to 0b01
		{
			tunevcocap_max	= i;
			break;
		}
	}
	tunevcocap_res	= (tunevcocap_max + tunevcocap_min) >> 1;	// (max+min) / 2
	printf("tunevcocap_tx_min=%d\r\n", tunevcocap_min);
	printf("tunevcocap_tx_max=%d\r\n", tunevcocap_max);
	printf("tunevcocap_tx_res=%d\r\n", tunevcocap_res);
	lime_spi_blk_write(cmdcontext_spi,	0x19, (reg0x19_val&0xC0)|tunevcocap_res );
	return tunevcocap_res;
}

unsigned char	tunevcocap_rx(void *cmdcontext_spi)
{
	unsigned char	retval			= 0xFF;
	unsigned char	reg0x29_val		= 0;
	unsigned char	reg0x2A_val		= 0;
	int				i				= 0;
	unsigned char	vtune			= 0xFF;
	unsigned char	tunevcocap_min	= 0xFF;
	unsigned char	tunevcocap_max	= 0x3F;
	unsigned char	tunevcocap_res	= 0xFF;
	unsigned char	min_is_done		= 0;
	lime_spi_blk_read(cmdcontext_spi,	0x29, &reg0x29_val);
	for (i = 0; i < 64; i++)
	{
		lime_spi_blk_write(cmdcontext_spi,	0x29, (reg0x29_val&0xC0)|i );
		lime_spi_blk_read(cmdcontext_spi,	0x2A, &reg0x2A_val);
		vtune	= (reg0x2A_val>>6)&0x03;
//		printf("i=%d vtune=0x%x reg0x2A_val=0x%.2x\r\n", i, vtune, reg0x2A_val);
		if ( (vtune == 0) && (!min_is_done) )	// pass 0b10 to 0b00
		{
			tunevcocap_min	= i;
			min_is_done		= 1;
		}
		else if (vtune == 1)	// pass 0b00 to 0b01
		{
			tunevcocap_max	= i;
			break;
		}
	}
	tunevcocap_res	= (tunevcocap_max + tunevcocap_min) >> 1;	// (max+min) / 2
	printf("tunevcocap_rx_min=%d\r\n", tunevcocap_min);
	printf("tunevcocap_rx_max=%d\r\n", tunevcocap_max);
	printf("tunevcocap_rx_res=%d\r\n", tunevcocap_res);
	lime_spi_blk_write(cmdcontext_spi,	0x29, (reg0x29_val&0xC0)|tunevcocap_res );
	return tunevcocap_res;
}

void bladerf_strerror(int error)
{
    switch (error) {
        case BLADERF_ERR_UNEXPECTED:
            printf("Calibration failed: An unexpected error occurred\r\n");
            break;
        case BLADERF_ERR_RANGE:
        	printf("Calibration failed: Provided parameter was out of the allowable range\r\n");
            break;
        case BLADERF_ERR_INVAL:
            printf("Calibration failed: Invalid operation or parameter\r\n");
            break;
        case BLADERF_ERR_MEM:
            printf("Calibration failed: A memory allocation error occurred\r\n");
            break;
        case BLADERF_ERR_IO:
            printf("Calibration failed: File or device I/O failure\r\n");
            break;
        case BLADERF_ERR_TIMEOUT:
            printf("Calibration failed: Operation timed out\r\n");
            break;
        case BLADERF_ERR_NODEV:
            printf("Calibration failed: No devices available\r\n");
            break;
        case BLADERF_ERR_UNSUPPORTED:
            printf("Calibration failed: Operation not supported\r\n");
            break;
        case BLADERF_ERR_MISALIGNED:
            printf("Calibration failed: Misaligned flash access\r\n");
            break;
        case BLADERF_ERR_CHECKSUM:
            printf("Calibration failed: Invalid checksum\r\n");
            break;
        case BLADERF_ERR_NO_FILE:
            printf("Calibration failed: File not found\r\n");
            break;
        case BLADERF_ERR_UPDATE_FPGA:
            printf("Calibration failed: An FPGA update is required\r\n");
            break;
        case BLADERF_ERR_UPDATE_FW:
            printf("Calibration failed: A firmware update is required\r\n");
            break;
        case BLADERF_ERR_TIME_PAST:
            printf("Calibration failed: Requested timestamp is in the past\r\n");
            break;
        case BLADERF_ERR_QUEUE_FULL:
            printf("Calibration failed: Could not enqueue data into full queue\r\n");
            break;
        case BLADERF_ERR_FPGA_OP:
            printf("Calibration failed: An FPGA operation reported a failure\r\n");
            break;
        case BLADERF_ERR_PERMISSION:
            printf("Calibration failed: Insufficient permissions for the requested operation\r\n");
            break;
        case BLADERF_ERR_WOULD_BLOCK:
            printf("Calibration failed: The operation would block, but has been requested to be non-blocking\r\n");
            break;
        case 0:
            printf("Calibration failed: Success\r\n");
            break;
        default:
            printf("Calibration failed: Unknown error code\r\n");
            break;
    }
    return;
}

/* Convert ms to samples */
#define MS_TO_SAMPLES(ms_, rate_) (\
    (unsigned int) (ms_ * ((uint64_t) rate_) / 1000) \
)

/*******************************************************************************
 * TX DC offset calibration
 ******************************************************************************/

#define TX_CAL_RATE     (4000000)

#define TX_CAL_RX_BW    (3000000)
#define TX_CAL_RX_LNA   (BLADERF_LNA_GAIN_MAX)
#define TX_CAL_RX_VGA1  (25)
#define TX_CAL_RX_VGA2  (0)

#define TX_CAL_TX_BW    (1500000)

#define TX_CAL_TS_INC   (MS_TO_SAMPLES(15, TX_CAL_RATE))
#define TX_CAL_COUNT    (MS_TO_SAMPLES(5,  TX_CAL_RATE))

#define TX_CAL_CORR_SWEEP_LEN (4096 / 16)   /* -2048:16:2048 */

#define TX_CAL_DEFAULT_LB (BLADERF_LB_RF_LNA1)

struct tx_cal_backup
{
	unsigned int rx_freq;
	struct bladerf_rational_rate rx_sample_rate;
	unsigned int rx_bandwidth;

	bladerf_lna_gain rx_lna;
	int rx_vga1;
	int rx_vga2;

	struct bladerf_rational_rate tx_sample_rate;
	unsigned int tx_bandwidth;

	bladerf_loopback loopback;
};

int tuning_get_freq(void *dev,
					bladerf_module module,
                    unsigned int *frequency)
{
	struct lms_freq f;
	int rv = 0;

	rv = lms_get_frequency( dev, module, &f );
	if (rv != 0) {
		return rv;
	}

	if( f.x == 0 )
	{
		/* If we see this, it's most often an indication that communication
		* with the LMS6002D is not occuring correctly */
		*frequency = 0 ;
		rv = BLADERF_ERR_IO;
	}
	else
	{
		*frequency = lms_frequency_to_hz(&f);
	}
		if (rv != 0) {
		return rv;
	}
//	rv = xb_get_attached(dev, &attached);
//	if (rv != 0) {
//	return rv;
//	}
//	if (attached == BLADERF_XB_200) {
//	rv = xb200_get_path(dev, module, &path);
//	if (rv != 0) {
//	return rv;
//	}
//	if (path == BLADERF_XB200_MIX) {
//	*frequency = 1248000000 - *frequency;
//	}
//	}
	return rv;
}
int bladerf_get_frequency(void *dev,
                            bladerf_module module, unsigned int *frequency)
{
	int status;
	printf("bladerf_get_frequency\r\n");
	status = tuning_get_freq(dev, module, frequency);
	return status;
}

/* LPF conversion table */
static const unsigned int uint_bandwidths[] = {
    MHz(28),
    MHz(20),
    MHz(14),
    MHz(12),
    MHz(10),
    kHz(8750),
    MHz(7),
    MHz(6),
    kHz(5500),
    MHz(5),
    kHz(3840),
    MHz(3),
    kHz(2750),
    kHz(2500),
    kHz(1750),
    kHz(1500)
};

lms_bw lms_uint2bw(unsigned int req)
{
    lms_bw ret;

    if (     req <= kHz(1500)) ret = BW_1p5MHz;
    else if (req <= kHz(1750)) ret = BW_1p75MHz;
    else if (req <= kHz(2500)) ret = BW_2p5MHz;
    else if (req <= kHz(2750)) ret = BW_2p75MHz;
    else if (req <= MHz(3)  )  ret = BW_3MHz;
    else if (req <= kHz(3840)) ret = BW_3p84MHz;
    else if (req <= MHz(5)  )  ret = BW_5MHz;
    else if (req <= kHz(5500)) ret = BW_5p5MHz;
    else if (req <= MHz(6)  )  ret = BW_6MHz;
    else if (req <= MHz(7)  )  ret = BW_7MHz;
    else if (req <= kHz(8750)) ret = BW_8p75MHz;
    else if (req <= MHz(10) )  ret = BW_10MHz;
    else if (req <= MHz(12) )  ret = BW_12MHz;
    else if (req <= MHz(14) )  ret = BW_14MHz;
    else if (req <= MHz(20) )  ret = BW_20MHz;
    else                       ret = BW_28MHz;

    return ret;
}

int lms_get_bandwidth(void *dev, bladerf_module mod, lms_bw *bw)
{
    int status;
    uint8_t data;
    const uint8_t reg = (mod == BLADERF_MODULE_RX) ? 0x54 : 0x34;

	printf("lms_get_bandwidth\r\n");

	status = LMS_READ(dev, reg, &data);
    if (status != 0) {
        return status;
    }

    /* Fetch bandwidth table index from reg[5:2] */
    data >>= 2;
    data &= 0xf;

    //assert(data < ARRAY_SIZE(uint_bandwidths));
    *bw = (lms_bw)data;
    return 0;
}

unsigned int lms_bw2uint(lms_bw bw)
{
    unsigned int idx = bw & 0xf;
	printf("lms_bw2uint\r\n");
    //assert(idx < ARRAY_SIZE(uint_bandwidths));
    return uint_bandwidths[idx];
}

int bladerf_get_bandwidth(void *dev, bladerf_module module,
                            unsigned int *bandwidth)
{
    int		status;
    lms_bw	bw;

	printf("bladerf_get_bandwidth\r\n");

	status = lms_get_bandwidth( dev, module, &bw);

    if (status == 0) {
        *bandwidth = lms_bw2uint(bw);
    } else {
        *bandwidth = 0;
    }

    return status;
}

int bladerf_get_lna_gain(void *dev, bladerf_lna_gain *gain)
{
    int status;
    status = lms_lna_get_gain(dev, gain);
    return status;
}

int bladerf_get_rxvga1(void *dev, int *gain)
{
    int status;
    status = lms_rxvga1_get_gain(dev, gain);
    return status;
}

int bladerf_get_rxvga2(void *dev, int *gain)
{
    int status;
    status = lms_rxvga2_get_gain(dev, gain);
    return status;
}

int bladerf_set_lna_gain(void *dev, bladerf_lna_gain gain)
{
    int status;
    status = lms_lna_set_gain(dev, gain);
    return status;
}

int bladerf_set_rxvga1(void *dev, int gain)
{
    int status;
    status = lms_rxvga1_set_gain(dev, gain);
    return status;
}

int bladerf_set_rxvga2(void *dev, int gain)
{
    int status;
    status = lms_rxvga2_set_gain(dev, gain);
    return status;
}

int lms_set_bandwidth(void *dev, bladerf_module mod, lms_bw bw)
{
    int status;
    uint8_t data;
    const uint8_t reg = (mod == BLADERF_MODULE_RX) ? 0x54 : 0x34;

    status = LMS_READ(dev, reg, &data);
    if (status != 0) {
        return status;
    }

    data &= ~0x3c;      /* Clear out previous bandwidth setting */
    data |= (bw << 2);  /* Apply new bandwidth setting */

    return LMS_WRITE(dev, reg, data);
}

int lms_lpf_enable(void *dev, bladerf_module mod, bool enable)
{
    int status;
    uint8_t data;
    const uint8_t reg = (mod == BLADERF_MODULE_RX) ? 0x54 : 0x34;

    status = LMS_READ(dev, reg, &data);
    if (status != 0) {
        return status;
    }

    if (enable) {
        data |= (1 << 1);
    } else {
        data &= ~(1 << 1);
    }

    status = LMS_WRITE(dev, reg, data);
    if (status != 0) {
        return status;
    }

    /* Check to see if we are bypassed */
    status = LMS_READ(dev, reg + 1, &data);
    if (status != 0) {
        return status;
    } else if (data & (1 << 6)) {
        /* Bypass is enabled; switch back to normal operation */
        data &= ~(1 << 6);
        status = LMS_WRITE(dev, reg + 1, data);
    }

    return status;
}

int bladerf_set_bandwidth(void *dev, bladerf_module module,
                          unsigned int bandwidth,
                          unsigned int *actual)
{
    int status;
    lms_bw bw;

	printf("bladerf_set_bandwidth\r\n");
    if (bandwidth < BLADERF_BANDWIDTH_MIN) {
        bandwidth = BLADERF_BANDWIDTH_MIN;
        printf("Clamping bandwidth to %dHz\r\n", bandwidth);
    } else if (bandwidth > BLADERF_BANDWIDTH_MAX) {
        bandwidth = BLADERF_BANDWIDTH_MAX;
        printf("Clamping bandwidth to %dHz\r\n", bandwidth);
    }

    bw = lms_uint2bw(bandwidth);

    status = lms_lpf_enable(dev, module, true);
    if (status != 0) {
        goto out;
    }

    status = lms_set_bandwidth(dev, module, bw);
    if (actual != NULL) {
        if (status == 0) {
            *actual = lms_bw2uint(bw);
        } else {
            *actual = 0;
        }
    }

out:
    return status;
}

static int get_tx_cal_backup(void *dev, struct tx_cal_backup *b)
{
    int status;

    status = bladerf_get_frequency(dev, BLADERF_MODULE_RX, &b->rx_freq);
    if (status != 0) {
        return status;
    }

//    status = bladerf_get_rational_sample_rate(dev, BLADERF_MODULE_RX,
//                                              &b->rx_sample_rate);
//    if (status != 0) {
//        return status;
//    }

    status = bladerf_get_bandwidth(dev, BLADERF_MODULE_RX, &b->rx_bandwidth);
    if (status != 0) {
        return status;
    }

    status = bladerf_get_lna_gain(dev, &b->rx_lna);
    if (status != 0) {
        return status;
    }

    status = bladerf_get_rxvga1(dev, &b->rx_vga1);
    if (status != 0) {
        return status;
    }

    status = bladerf_get_rxvga2(dev, &b->rx_vga2);
    if (status != 0) {
        return status;
    }

//    status = bladerf_get_rational_sample_rate(dev, BLADERF_MODULE_TX,
//                                              &b->tx_sample_rate);
//    if (status != 0) {
//        return status;
//    }

    status = bladerf_get_loopback(dev, &b->loopback);

    return status;
}

struct complexf {
    float i;
    float q;
};

struct tx_cal {
    struct bladerf *dev;
    int16_t *samples;           /* Raw samples */
    unsigned int num_samples;   /* Number of raw samples */
    struct complexf *filt;      /* Filter state */
    struct complexf *filt_out;  /* Filter output */
    struct complexf *post_mix;  /* Post-filter, mixed to baseband */
    int16_t *sweep;             /* Correction sweep */
    float   *mag;               /* Magnitude results from sweep */
    uint64_t ts;                /* Timestamp */
    bladerf_loopback loopback;  /* Current loopback mode */
    bool rx_low;                /* RX tuned lower than TX */
};

static int apply_tx_cal_settings(void *dev)
{
    int status;

//    status = bladerf_set_sample_rate(dev, BLADERF_MODULE_RX, TX_CAL_RATE, NULL);
//    if (status != 0) {
//        return status;
//    }

    status = bladerf_set_bandwidth(dev, BLADERF_MODULE_RX, TX_CAL_RX_BW, NULL);
    if (status != 0) {
        return status;
    }

    status = bladerf_set_lna_gain(dev, TX_CAL_RX_LNA);
    if (status != 0) {
        return status;
    }

    status = bladerf_set_rxvga1(dev, TX_CAL_RX_VGA1);
    if (status != 0) {
        return status;
    }

    status = bladerf_set_rxvga2(dev, TX_CAL_RX_VGA2);
    if (status != 0) {
        return status;
    }

//    status = bladerf_set_sample_rate(dev, BLADERF_MODULE_TX, TX_CAL_RATE, NULL);
//    if (status != 0) {
//        return status;
//    }

//    status = bladerf_set_loopback(dev, TX_CAL_DEFAULT_LB);	// TX_CAL_DEFAULT_LB = BLADERF_LB_RF_LNA1
    status = bladerf_set_loopback(dev, BLADERF_LB_RF_LNA2);
	if (status != 0) {
        return status;
    }

    return status;
}

/* We just need to flush some zeros through the system to hole the DAC at
 * 0+0j and remain there while letting it underrun. This alleviates the
 * need to worry about continuously TX'ing zeros. */
static int tx_cal_tx_init(void *dev)
{
    int status;
    int16_t zero_sample[] = { 0, 0 };
    struct bladerf_metadata meta;

    memset(&meta, 0, sizeof(meta));

    /* TODO : run samples with I=0 and Q=0 */
    // Already done with cal_tx_lpf_init()



//    status = bladerf_sync_config(dev, BLADERF_MODULE_TX,
//                                 BLADERF_FORMAT_SC16_Q11_META,
//                                 4, 16384, 2, 1000);
//    if (status != 0) {
//        return status;
//    }

    status = bladerf_enable_module(dev, BLADERF_MODULE_TX, true);
    if (status != 0) {
        return status;
    }

    meta.flags = BLADERF_META_FLAG_TX_BURST_START |
                 BLADERF_META_FLAG_TX_BURST_END   |
                 BLADERF_META_FLAG_TX_NOW;

//    status = bladerf_sync_tx(dev, &zero_sample, 1, &meta, 2000);
    return status;
}

static int tx_cal_rx_init(void *dev)
{
    int status;

//    status = bladerf_sync_config(dev, BLADERF_MODULE_RX,
//                                 BLADERF_FORMAT_SC16_Q11_META,
//                                 64, 16384, 32, 1000);
//    if (status != 0) {
//        return status;
//    }

    status = bladerf_enable_module(dev, BLADERF_MODULE_RX, true);
    return status;
}

/* Filter used to isolate contribution of TX LO leakage in received
 * signal. 15th order Equiripple FIR with Fs=4e6, Fpass=1, Fstop=1e6
 */
static const float tx_cal_filt[] = {
    0.000327949366768f, 0.002460188536582f, 0.009842382390924f,
    0.027274728394777f, 0.057835200476419f, 0.098632713294830f,
    0.139062540460741f, 0.164562494987592f, 0.164562494987592f,
    0.139062540460741f, 0.098632713294830f, 0.057835200476419f,
    0.027274728394777f, 0.009842382390924f, 0.002460188536582f,
    0.000327949366768f,
};

static const unsigned int tx_cal_filt_num_taps =
    (sizeof(tx_cal_filt) / sizeof(tx_cal_filt[0]));

/*------------------------------------------------------------------------------
 * Get current timestamp counter
 *----------------------------------------------------------------------------*/
int bladerf_get_timestamp(void *dev, bladerf_module module, uint64_t *value)
{
    int	status	 = 0;
//    status = dev->fn->get_timestamp(dev,module,value);
    return status;
}

/* This should be called immediately preceding the cal routines */
static int tx_cal_state_init(void *dev, struct tx_cal *cal)
{
    int status;

    cal->dev = dev;
//    cal->num_samples = TX_CAL_COUNT;
//    cal->loopback = TX_CAL_DEFAULT_LB;
    cal->num_samples = 76800;
    cal->loopback = BLADERF_LB_RF_LNA2;

    /* Interleaved SC16 Q11 samples */
//    cal->samples = malloc(2 * sizeof(cal->samples[0]) * cal->num_samples);
//    if (cal->samples == NULL) {
//        return BLADERF_ERR_MEM;
//    }
//    cal->samples	=

    /* Filter state */
    cal->filt = malloc(2 * sizeof(cal->filt[0]) * tx_cal_filt_num_taps);
    if (cal->filt == NULL) {
        return BLADERF_ERR_MEM;
    }

    /* Filter output */
    cal->filt_out = malloc(sizeof(cal->filt_out[0]) * cal->num_samples);
    if (cal->filt_out == NULL) {
        return BLADERF_ERR_MEM;
    }

    /* Post-mix */
    cal->post_mix = malloc(sizeof(cal->post_mix[0]) * cal->num_samples);
    if (cal->post_mix == NULL) {
        return BLADERF_ERR_MEM;
    }

    /* Correction sweep and results */
    cal->sweep = malloc(sizeof(cal->sweep[0]) * TX_CAL_CORR_SWEEP_LEN);
    if (cal->sweep == NULL) {
        return BLADERF_ERR_MEM;
    }

    cal->mag = malloc(sizeof(cal->mag[0]) * TX_CAL_CORR_SWEEP_LEN);
    if (cal->mag == NULL) {
        return BLADERF_ERR_MEM;
    }

    /* Set initial RX in the future */
//    status = bladerf_get_timestamp(cal->dev, BLADERF_MODULE_RX, &cal->ts);
//    if (status == 0) {
//        cal->ts += 20 * TX_CAL_TS_INC;
//    }

    return status;
}

int bladerf_set_frequency(void *dev,
                          bladerf_module module, unsigned int frequency)
{
    int status;
	printf("bladerf_set_frequency\r\n");
    status = tuning_set_freq(dev, module, frequency);
    return status;
}

static int tx_cal_update_frequency(struct tx_cal *state, unsigned int freq)
{
    int status;
    bladerf_loopback lb;
    unsigned int rx_freq;

    status = bladerf_set_frequency(state->dev, BLADERF_MODULE_TX, freq);
    if (status != 0) {
        return status;
    }

    rx_freq = freq - 1920000;	// fs/4
    if (rx_freq < BLADERF_FREQUENCY_MIN) {
        rx_freq = freq + 1920000;
        state->rx_low = false;
    } else {
        state->rx_low = true;
    }

    status = bladerf_set_frequency(state->dev, BLADERF_MODULE_RX, rx_freq);
    if (status != 0) {
        return status;
    }

    if (freq < 1500000000) {
        lb = BLADERF_LB_RF_LNA1;
        printf("Switching to RF LNA1 loopback.\r\n");
    } else {
        lb = BLADERF_LB_RF_LNA2;
        printf("Switching to RF LNA2 loopback.\r\n");
    }

    if (state->loopback != lb) {
        status = bladerf_set_loopback(state->dev, lb);
        if (status == 0) {
            state->loopback = lb;
        }
    }

    return status;
}

/* RX samples, retrying if the machine is struggling to keep up. */
static int rx_samples(void *dev, int16_t *samples,
                      unsigned int count, uint64_t *ts, uint64_t ts_inc)
{
    int						status		= 0;
    struct bladerf_metadata	meta;
    int						retry		= 0;
    const int				max_retries	= 10;
    bool					overrun		= true;

	int				card		= 0;
	int				ant			= 0;
	unsigned int	*p_rx_ant0	= NULL;
	unsigned int	*p_tx_ant0	= NULL;

	printf("rx_samples\r\n");
    memset(&meta, 0, sizeof(meta));
    meta.timestamp = *ts;

    /* TODO : RX sample capture */
    get_rx_samples(card);

	// pointer to data
	// p_rx_ant0 = openair0_exmimo_pci[ card ].adc_head[ ant ];
//	p_rx_ant0 = openair0_exmimo_pci[ card ].adc_head[ ant ];
//	p_tx_ant0 = openair0_exmimo_pci[ card ].dac_head[ ant ];

	samples	= (uint16_t *)openair0_exmimo_pci[ card ].adc_head[ ant ];
	printf("rx_samples : samples=0x%08lX\r\n", (long unsigned int)samples);

#if 0
	while (status == 0 && overrun && retry < max_retries)
	{
		meta.timestamp	= *ts;
		status			= bladerf_sync_rx(dev, samples, count, &meta, 2000);

		if (status == BLADERF_ERR_TIME_PAST)
		{
			status = bladerf_get_timestamp(dev, BLADERF_MODULE_RX, ts);
			if (status != 0)
			{
				return status;
			}
			else
			{
				*ts		+= 20 * ts_inc;
				retry++;
				status	= 0;
			}
		}
		else if (status == 0)
		{
			overrun = (meta.flags & BLADERF_META_STATUS_OVERRUN) != 0;
			if (overrun)
			{
				*ts	+= count + ts_inc;
				retry++;
			}
		}
		else
		{
			return status;
		}
	}
	if (retry >= max_retries)
	{
		status	= BLADERF_ERR_IO;
	}
	else if (status == 0)
	{
		*ts	+= count + ts_inc;
	}
#endif
    return status;
}

/* Filter samples
 *  Input:  state->post_mix
 *  Output: state->filt_out
 */
static void tx_cal_filter(struct tx_cal *state)
{
    unsigned int n, m;
    struct complexf *ins1, *ins2;
    struct complexf *curr; /* Current filter state */
    const struct complexf *filt_end = &state->filt[2 * tx_cal_filt_num_taps];

    /* Reset filter state */
    ins1 = &state->filt[0];
    ins2 = &state->filt[tx_cal_filt_num_taps];
    memset(state->filt, 0, 2 * sizeof(state->filt[0]) * tx_cal_filt_num_taps);

    for (n = 0; n < state->num_samples; n++) {
        /* Insert sample */
        *ins1 = *ins2 = state->post_mix[n];

        /* Convolve */
        state->filt_out[n].i = 0;
        state->filt_out[n].q = 0;
        curr = ins2;

        for (m = 0; m < tx_cal_filt_num_taps; m++, curr--) {
            state->filt_out[n].i += tx_cal_filt[m] * curr->i;
            state->filt_out[n].q += tx_cal_filt[m] * curr->q;
        }

        /* Update insertion points */
        ins2++;
        if (ins2 == filt_end) {
            ins1 = &state->filt[0];
            ins2 = &state->filt[tx_cal_filt_num_taps];
        } else {
            ins1++;
        }

    }
}

/* Deinterleave, scale, and mix with an -Fs/4 tone to shift TX DC offset out at
 * Fs/4 to baseband.
 *  Input:  state->samples
 *  Output: state->post_mix
 */
static void tx_cal_mix(struct tx_cal *state)
{
    unsigned int n, m;
    int mix_state;
    float scaled_i, scaled_q;

    /* Mix with -Fs/4 if RX is tuned "lower" than TX, and Fs/4 otherwise */
    const int mix_state_inc = state->rx_low ? 1 : -1;
    mix_state = 0;

    for (n = 0, m = 0; n < (2 * state->num_samples); n += 2, m++) {
        scaled_i = state->samples[n]   / 2048.0f;
        scaled_q = state->samples[n+1] / 2048.0f;

        switch (mix_state) {
            case 0:
                state->post_mix[m].i =  scaled_i;
                state->post_mix[m].q =  scaled_q;
                break;

            case 1:
                state->post_mix[m].i =  scaled_q;
                state->post_mix[m].q = -scaled_i;
                break;

            case 2:
                state->post_mix[m].i = -scaled_i;
                state->post_mix[m].q = -scaled_q;
                break;

            case 3:
                state->post_mix[m].i = -scaled_q;
                state->post_mix[m].q =  scaled_i;
                break;
        }

        mix_state = (mix_state + mix_state_inc) & 0x3;
    }
}

static int tx_cal_avg_magnitude(struct tx_cal *state, float *avg_mag)
{
    int status;
    const unsigned int start = (tx_cal_filt_num_taps + 1) / 2;
    unsigned int n;
    float accum;

    /* Fetch samples at the current settings */
    status = rx_samples(state->dev, state->samples, state->num_samples,
                        &state->ts, TX_CAL_TS_INC);
    if (status != 0) {
        return status;
    }

    /* Deinterleave & mix TX's DC offset contribution to baseband */
    tx_cal_mix(state);

    /* Filter out everything other than the TX DC offset's contribution */
    tx_cal_filter(state);

    /* Compute the power (magnitude^2 to alleviate need for square root).
     * We skip samples here to account for the group delay of the filter;
     * the initial samples will be ramping up. */
    accum = 0;
    for (n = start; n < state->num_samples; n++) {
        const struct complexf *s = &state->filt_out[n];
        const float m = (float) sqrt(s->i * s->i + s->q * s->q);
        accum += m;
    }

    *avg_mag = (accum / (state->num_samples - start));

    /* Scale this back up to DAC/ADC counts, just for convenience */
    *avg_mag *= 2048.0;

    return status;
}

static inline uint8_t scale_dc_offset(bladerf_module module, int16_t value)
{
    uint8_t ret;

    switch (module) {
        case BLADERF_MODULE_RX:
            /* RX only has 6 bits of scale to work with, remove normalization */
            value >>= 5;

            if (value < 0) {
                if (value <= -64) {
                    /* Clamp */
                    value = 0x3f;
                } else {
                    value = (-value) & 0x3f;
                }

                /* This register uses bit 6 to denote a negative value */
                value |= (1 << 6);
            } else {
                if (value >= 64) {
                    /* Clamp */
                    value = 0x3f;
                } else {
                    value = value & 0x3f;
                }
            }

            ret = (uint8_t) value;
            break;

        case BLADERF_MODULE_TX:
            /* TX only has 7 bits of scale to work with, remove normalization */
            value >>= 4;

            /* LMS6002D 0x00 = -16, 0x80 = 0, 0xff = 15.9375 */
            if (value >= 0) {
                ret = (uint8_t) (value >= 128) ? 0x7f : (value & 0x7f);

                /* Assert bit 7 for positive numbers */
                ret = (1 << 7) | ret;
            } else {
                ret = (uint8_t) (value <= -128) ? 0x00 : (value & 0x7f);
            }
            break;

        default:
            //assert(!"Invalid module provided");
            ret = 0x00;
    }

    return ret;
}

static int set_dc_offset_reg(void *dev, bladerf_module module,
                             uint8_t addr, int16_t value)
{
    int status;
    uint8_t regval, tmp;

    switch (module) {
        case BLADERF_MODULE_RX:
            status = LMS_READ(dev, addr, &tmp);
            if (status != 0) {
                return status;
            }

            /* Bit 7 is unrelated to lms dc correction, save its state */
            tmp = tmp & (1 << 7);
            regval = scale_dc_offset(module, value) | tmp;
            break;

        case BLADERF_MODULE_TX:
            regval = scale_dc_offset(module, value);
            break;

        default:
            return BLADERF_ERR_INVAL;
    }

    status = LMS_WRITE(dev, addr, regval);
    return status;
}

int lms_set_dc_offset_i(void *dev,
                        bladerf_module module, uint16_t value)
{
    const uint8_t addr = (module == BLADERF_MODULE_TX) ? 0x42 : 0x71;
    return set_dc_offset_reg(dev, module, addr, value);
}

int lms_set_dc_offset_q(void *dev,
                        bladerf_module module, int16_t value)
{
    const uint8_t addr = (module == BLADERF_MODULE_TX) ? 0x43 : 0x72;
    return set_dc_offset_reg(dev, module, addr, value);
}

/*------------------------------------------------------------------------------
 * IQ Calibration routines
 *----------------------------------------------------------------------------*/
int bladerf_set_correction(void *dev, bladerf_module module,
                           bladerf_correction corr, int16_t value)
{
    int status;
    switch (corr) {
//        case BLADERF_CORR_FPGA_PHASE:
//            status = dev->fn->set_iq_phase_correction(dev, module, value);
//            break;
//
//        case BLADERF_CORR_FPGA_GAIN:
//            /* Gain correction requires than an offset be applied */
//            value += (int16_t) 4096;
//            status = dev->fn->set_iq_gain_correction(dev, module, value);
//            break;
//
        case BLADERF_CORR_LMS_DCOFF_I:
            status = lms_set_dc_offset_i(dev, module, value);
            break;

        case BLADERF_CORR_LMS_DCOFF_Q:
            status = lms_set_dc_offset_q(dev, module, value);
            break;

        default:
            status = BLADERF_ERR_INVAL;
            //log_debug("Invalid correction type: %d\r\n", corr);
            break;
    }
    return status;
}

int get_dc_offset(struct bladerf *dev, bladerf_module module,
                  uint8_t addr, int16_t *value)
{
    int status;
    uint8_t tmp;

    status = LMS_READ(dev, addr, &tmp);
    if (status != 0) {
        return status;
    }

    switch (module) {
        case BLADERF_MODULE_RX:

            /* Mask out an unrelated control bit */
            tmp = tmp & 0x7f;

            /* Determine sign */
            if (tmp & (1 << 6)) {
                *value = -(int16_t)(tmp & 0x3f);
            } else {
                *value = (int16_t)(tmp & 0x3f);
            }

            /* Renormalize to 2048 */
            *value <<= 5;
            break;

        case BLADERF_MODULE_TX:
            *value = (int16_t) tmp;

            /* Renormalize to 2048 */
            *value <<= 4;
            break;

        default:
            return BLADERF_ERR_INVAL;
    }

    return 0;
}

int lms_get_dc_offset_i(struct bladerf *dev,
                        bladerf_module module, int16_t *value)
{
    const uint8_t addr = (module == BLADERF_MODULE_TX) ? 0x42 : 0x71;
    return get_dc_offset(dev, module, addr, value);
}

int lms_get_dc_offset_q(struct bladerf *dev,
                        bladerf_module module, int16_t *value)
{
    const uint8_t addr = (module == BLADERF_MODULE_TX) ? 0x43 : 0x72;
    return get_dc_offset(dev, module, addr, value);
}

int bladerf_get_correction(void *dev, bladerf_module module,
                           bladerf_correction corr, int16_t *value)
{
    int status;
    switch (corr) {
//        case BLADERF_CORR_FPGA_PHASE:
//            status = dev->fn->get_iq_phase_correction(dev, module, value);
//            break;
//
//        case BLADERF_CORR_FPGA_GAIN:
//            status = dev->fn->get_iq_gain_correction(dev, module, value);
//
//            /* Undo the gain control offset */
//            if (status == 0) {
//                *value -= 4096;
//            }
//            break;
//
        case BLADERF_CORR_LMS_DCOFF_I:
            status = lms_get_dc_offset_i(dev, module, value);
            break;

        case BLADERF_CORR_LMS_DCOFF_Q:
            status = lms_get_dc_offset_q(dev, module, value);
            break;

        default:
            status = BLADERF_ERR_INVAL;
            //log_debug("Invalid correction type: %d\r\n", corr);
            break;
    }
    return status;
}

/* Apply the correction value and read the TX DC offset magnitude */
static int tx_cal_measure_correction(struct tx_cal *state,
                                     bladerf_correction c,
                                     int16_t value, float *mag)
{
    int status;

    status = bladerf_set_correction(state->dev, BLADERF_MODULE_TX, c, value);
    if (status != 0) {
        return status;
    }

    state->ts += TX_CAL_TS_INC;

    status = tx_cal_avg_magnitude(state, mag);
    if (status == 0) {
        printf("  Corr=%5d, Avg_magnitude=%f\r\n", value, *mag);
    }

    return status;
}

static int tx_cal_get_corr(struct tx_cal *state, bool i_ch,
                           int16_t *corr_value, float *error_value)
{
    int status;
    unsigned int n;
    int16_t corr;
    float mag[4];
    float m1, m2, b1, b2;
    int16_t range_min, range_max;
    int16_t min_corr;
    float   min_mag;

    const int16_t x[4] = { -1800, -1000, 1000, 1800 };

    const bladerf_correction corr_module =
        i_ch ? BLADERF_CORR_LMS_DCOFF_I : BLADERF_CORR_LMS_DCOFF_Q;

    printf("Getting coarse estimate for %c\r\n", i_ch ? 'I' : 'Q');

    for (n = 0; n < 4; n++) {
        status = tx_cal_measure_correction(state, corr_module, x[n], &mag[n]);
        if (status != 0) {
            return status;
        }

    }

    m1 = (mag[1] - mag[0]) / (x[1] - x[0]);
    b1 = mag[0] - m1 * x[0];

    m2 = (mag[3] - mag[2]) / (x[3] - x[2]);
    b2 = mag[2] - m2 * x[2];

    printf("  m1=%3.8f, b1=%3.8f, m2=%3.8f, b=%3.8f\r\n", m1, b1, m2, b2);

    if (m1 < 0 && m2 > 0) {
        const int16_t tmp = (int16_t)((b2 - b1) / (m1 - m2) + 0.5);
        const int16_t corr_est = (tmp / 16) * 16;

        /* Number of points to sweep on either side of our estimate */
        const unsigned int n_sweep = 10;

        printf("  corr_est=%d\r\n", corr_est);

        range_min = corr_est - 16 * n_sweep;
        if (range_min < -2048) {
            range_min = -2048;
        }

        range_max = corr_est + 16 * n_sweep;
        if (range_max > 2048) {
            range_max = 2048;
        }

    } else {
        /* The frequency and gain combination have yielded a set of
         * points that do not form intersecting lines. This may be indicative
         * of a case where the LMS6 DC bias settings can't pull the DC offset
         * to a zero-crossing.  We'll just do a slow, full scan to find
         * a minimum */
        printf("  Could not compute estimate. Performing full sweep.\r\n");
        range_min = -2048;
        range_max = 2048;
    }


    printf("Performing correction value sweep: [%-5d : 16 :%5d]\r\n",
           range_min, range_max);

    min_corr = 0;
    min_mag  = 2048;

    for (n = 0, corr = range_min;
         corr <= range_max && n < TX_CAL_CORR_SWEEP_LEN;
         n++, corr += 16) {

        float tmp;

        status = tx_cal_measure_correction(state, corr_module, corr, &tmp);
        if (status != 0) {
            return status;
        }

        if (tmp < 0) {
            tmp = -tmp;
        }

        if (tmp < min_mag) {
            min_corr = corr;
            min_mag  = tmp;
        }
    }

    /* Leave the device set to the minimum */
    status = bladerf_set_correction(state->dev, BLADERF_MODULE_TX,
                                    corr_module, min_corr);
    if (status == 0) {
        *corr_value  = min_corr;
        *error_value = min_mag;
    }

    return status;
}

static int perform_tx_cal(struct tx_cal *state, struct dc_calibration_params *p)
{
    int status = 0;

    status = tx_cal_update_frequency(state, p->frequency);
    if (status != 0) {
        return status;
    }

    state->ts += TX_CAL_TS_INC;

    /* Perform I calibration */
    status = tx_cal_get_corr(state, true, &p->corr_i, &p->error_i);
    if (status != 0) {
        return status;
    }

    /* Perform Q calibration */
    status = tx_cal_get_corr(state, false, &p->corr_q, &p->error_q);
    if (status != 0) {
        return status;
    }

    /* Re-do I calibration to try to further fine-tune result */
    status = tx_cal_get_corr(state, true, &p->corr_i, &p->error_i);
    if (status != 0) {
        return status;
    }

    return status;
}

static void tx_cal_state_deinit(struct tx_cal *cal)
{
    free(cal->sweep);
    free(cal->mag);
//    free(cal->samples);
    free(cal->filt);
    free(cal->filt_out);
    free(cal->post_mix);
}

static int set_tx_cal_backup(struct bladerf *dev, struct tx_cal_backup *b)
{
    int status;
    int retval = 0;

    status = bladerf_set_loopback(dev, b->loopback);
    if (status != 0 && retval == 0) {
        retval = status;
    }

    status = bladerf_set_frequency(dev, BLADERF_MODULE_RX, b->rx_freq);
    if (status != 0 && retval == 0) {
        retval = status;
    }

//    status = bladerf_set_rational_sample_rate(dev, BLADERF_MODULE_RX,
//                                              &b->rx_sample_rate, NULL);
//    if (status != 0 && retval == 0) {
//        retval = status;
//    }

    status = bladerf_set_bandwidth(dev, BLADERF_MODULE_RX,
                                   b->rx_bandwidth, NULL);
    if (status != 0 && retval == 0) {
        retval = status;
    }

    status = bladerf_set_lna_gain(dev, b->rx_lna);
    if (status != 0 && retval == 0) {
        retval = status;
    }

    status = bladerf_set_rxvga1(dev, b->rx_vga1);
    if (status != 0 && retval == 0) {
        retval = status;
    }

    status = bladerf_set_rxvga2(dev, b->rx_vga2);
    if (status != 0 && retval == 0) {
        retval = status;
    }

//    status = bladerf_set_rational_sample_rate(dev, BLADERF_MODULE_TX,
//                                              &b->tx_sample_rate, NULL);
//    if (status != 0 && retval == 0) {
//        retval = status;
//    }

    return retval;
}

int dc_calibration_tx(void							*dev,
                      struct dc_calibration_params	*params,
                      size_t						num_params,
					  bool							print_status)
{
    int status = 0;
    int retval = 0;
    struct tx_cal_backup backup;
    struct tx_cal state;
    size_t i;

    memset(&state, 0, sizeof(state));

    /* Backup the device state prior to making changes */
    status = get_tx_cal_backup(dev, &backup);
    if (status != 0) {
        return status;
    }

    /* Configure the device for our TX cal operation */
    status = apply_tx_cal_settings(dev);
    if (status != 0) {
        goto out;
    }

    /* Enable TX and run zero samples through the device */
    status = tx_cal_tx_init(dev);
    if (status != 0) {
        goto out;
    }

    /* Enable RX */
    status = tx_cal_rx_init(dev);
    if (status != 0) {
        goto out;
    }

    /* Initialize calibration state information and resources */
    status = tx_cal_state_init(dev, &state);
    if (status != 0) {
        goto out;
    }

    for (i = 0; i < num_params && status == 0; i++) {
        status = perform_tx_cal(&state, &params[i]);

        if (status == 0 && print_status) {
#           ifdef DEBUG_DC_CALIBRATION
            const char sol = '\n';
            const char eol = '\n';
#           else
            const char sol = '\r';
            const char eol = '\0';
#           endif
            printf("%cCalibrated @ %10u Hz: "
                   "I=%4d (Error: %4.2f), "
                   "Q=%4d (Error: %4.2f)      %c",
                   sol,
                   params[i].frequency,
                   params[i].corr_i, params[i].error_i,
                   params[i].corr_q, params[i].error_q,
                   eol);
            fflush(stdout);
        }
    }

    if (print_status) {
        putchar('\n');
    }

out:
    retval = status;

    status = bladerf_enable_module(dev, BLADERF_MODULE_RX, false);
    if (status != 0 && retval == 0) {
        retval = status;
    }

    status = bladerf_enable_module(dev, BLADERF_MODULE_TX, false);
    if (status != 0 && retval == 0) {
        retval = status;
    }

    tx_cal_state_deinit(&state);

    status = set_tx_cal_backup(dev, &backup);
    if (status != 0 && retval == 0) {
        retval = status;
    }

    return retval;
}

/*******************************************************************************
 * RX DC offset calibration
 ******************************************************************************/

#define RX_CAL_RATE             (3000000)
#define RX_CAL_BW               (1500000)
#define RX_CAL_TS_INC           (MS_TO_SAMPLES(15, RX_CAL_RATE))
#define RX_CAL_COUNT            (MS_TO_SAMPLES(5,  RX_CAL_RATE))

#define RX_CAL_MAX_SWEEP_LEN    (2 * 2048 / 32) /* -2048 : 32 : 2048 */

struct rx_cal {
    struct bladerf *dev;

    int16_t *samples;
    unsigned int num_samples;

    int16_t *corr_sweep;

    uint64_t ts;

    unsigned int tx_freq;
};

struct rx_cal_backup {
    struct bladerf_rational_rate rational_sample_rate;
    unsigned int bandwidth;
    unsigned int tx_freq;
};

static int get_rx_cal_backup(void *dev, struct rx_cal_backup *b)
{
    int status;

	printf("get_rx_cal_backup\r\n");

//    status = bladerf_get_rational_sample_rate(dev, BLADERF_MODULE_RX,
//                                              &b->rational_sample_rate);
//    if (status != 0) {
//        return status;
//    }

    status = bladerf_get_bandwidth(dev, BLADERF_MODULE_RX, &b->bandwidth);
    if (status != 0) {
        return status;
    }

    status = bladerf_get_frequency(dev, BLADERF_MODULE_TX, &b->tx_freq);
    if (status != 0) {
        return status;
    }

    return status;
}

/* Ensure TX >= 1 MHz away from the RX frequency to avoid any potential
 * artifacts from the PLLs interfering with one another */
static int rx_cal_update_frequency(struct rx_cal *cal, unsigned int rx_freq)
{
    int status = 0;
    unsigned int f_diff;

	printf("rx_cal_update_frequency\r\n");

	if (rx_freq < cal->tx_freq) {
        f_diff = cal->tx_freq - rx_freq;
    } else {
        f_diff = rx_freq - cal->tx_freq;
    }

    printf("Set F_RX = %u\r\n", rx_freq);
    printf("F_diff(RX, TX) = %u\r\n", f_diff);

    if (f_diff < 1000000) {
        if (rx_freq >= (BLADERF_FREQUENCY_MIN + 1000000)) {
            cal->tx_freq = rx_freq - 1000000;
        } else {
            cal->tx_freq = rx_freq + 1000000;
        }

        status = bladerf_set_frequency(cal->dev, BLADERF_MODULE_TX,
                                       cal->tx_freq);
        if (status != 0) {
            return status;
        }

        printf("Adjusted TX frequency: %u\r\n", cal->tx_freq);
    }

    status = bladerf_set_frequency(cal->dev, BLADERF_MODULE_RX, rx_freq);
    if (status != 0) {
        return status;
    }

    cal->ts += RX_CAL_TS_INC;

    return status;
}

static inline void sample_mean(int16_t *samples, size_t count,
                               float *mean_i, float *mean_q)
{
    int64_t accum_i = 0;
    int64_t accum_q = 0;

    size_t n;


//	printf("sample_mean(count=%ld)\r\n", count);
    if (count == 0) {
        //assert(!"Invalid count (0) provided to sample_mean()");
        *mean_i = 0;
        *mean_q = 0;
        return;
    }

//	printf("sample_mean 2\r\n");

    for (n = 0; n < (2 * count); n += 2) {
        accum_i += samples[n];
        accum_q += samples[n + 1];
    }
//	printf("sample_mean 3\r\n");

    *mean_i = ((float) accum_i) / count;
    *mean_q = ((float) accum_q) / count;
}

static inline int set_rx_dc_corr(struct bladerf *dev, int16_t i, int16_t q)
{
    int status;

	printf("set_rx_dc_corr\r\n");
    status = bladerf_set_correction(dev, BLADERF_MODULE_RX,
                                    BLADERF_CORR_LMS_DCOFF_I, i);
    if (status != 0) {
        return status;
    }

    status = bladerf_set_correction(dev, BLADERF_MODULE_RX,
                                    BLADERF_CORR_LMS_DCOFF_Q, q);
    return status;
}

/* Get the mean for one of the coarse estimate points. If it seems that this
 * value might be (or close) causing us to clamp, adjust it and retry */
static int rx_cal_coarse_means(struct rx_cal *cal, int16_t *corr_value,
                               float *mean_i, float *mean_q)
{
    int status;
    const int16_t mean_limit_high = 2000;
    const int16_t mean_limit_low  = -mean_limit_high;
    const int16_t corr_limit = 128;
    bool retry = false;

    printf("rx_cal_coarse_means\r\n");
    do {
        status = set_rx_dc_corr(cal->dev, *corr_value, *corr_value);
        if (status != 0) {
            return status;
        }

        status = rx_samples(cal->dev, cal->samples, cal->num_samples,
                            &cal->ts, RX_CAL_TS_INC);
        if (status != 0) {
            return status;
        }

 		printf("rx_cal_coarse_means cal->samples @ = 0x%08lX\r\n", (long unsigned int)(cal->samples));

       sample_mean(cal->samples, cal->num_samples, mean_i, mean_q);

        if (*mean_i > mean_limit_high || *mean_q > mean_limit_high ||
            *mean_i < mean_limit_low  || *mean_q < mean_limit_low    ) {

            if (*corr_value < 0) {
                retry = (*corr_value <= -corr_limit);
            } else {
                retry = (*corr_value >= corr_limit);
            }

            if (retry) {
            	printf("Coarse estimate point Corr=%4d yields extreme means: "
                       "(%4f, %4f). Retrying...\r\n",
                       *corr_value, *mean_i, *mean_q);

                *corr_value = *corr_value / 2;
            }
        } else {
            retry = false;
        }
    } while (retry);

    if (retry) {
    	printf("Non-ideal values are being used.\r\n");
    }

    return 0;
}

/* Estimate the DC correction values that yield zero DC offset via a linear
 * approximation */
static int rx_cal_coarse_estimate(struct rx_cal *cal,
                                  int16_t *i_est, int16_t *q_est)
{
    int status;
    int16_t x1 = -2048;
    int16_t x2 = 2048;
    float y1i, y1q, y2i, y2q;
    float mi, mq;
    float bi, bq;
    float i_guess, q_guess;

    printf("rx_cal_coarse_estimate\r\n");

    status = rx_cal_coarse_means(cal, &x1, &y1i, &y1q);
    if (status != 0) {
        *i_est = 0;
        *q_est = 0;
        return status;
    }

    printf("Means for x1=%d: y1i=%f, y1q=%f\r\n", x1, y1i, y1q);

    status = rx_cal_coarse_means(cal, &x2, &y2i, &y2q);
    if (status != 0) {
        *i_est = 0;
        *q_est = 0;
        return status;
    }

    printf("Means for x2: y2i=%f, y2q=%f\r\n", y2i, y2q);

    mi = (y2i - y1i) / (x2 - x1);
    mq = (y2q - y1q) / (x2 - x1);

    bi = y1i - mi * x1;
    bq = y1q - mq * x1;

    printf("mi=%f, bi=%f, mq=%f, bq=%f\r\n", mi, bi, mq, bq);

    i_guess = -bi/mi + 0.5f;
    if (i_guess < -2048) {
        i_guess = -2048;
    } else if (i_guess > 2048) {
        i_guess = 2048;
    }

    q_guess = -bq/mq + 0.5f;
    if (q_guess < -2048) {
        q_guess = -2048;
    } else if (q_guess > 2048) {
        q_guess = 2048;
    }

    *i_est = (int16_t) i_guess;
    *q_est = (int16_t) q_guess;

    printf("Coarse estimate: I=%d, Q=%d\r\n", *i_est, *q_est);

    return 0;
}

static void init_rx_cal_sweep(int16_t *corr, unsigned int *sweep_len,
                              int16_t i_est, int16_t q_est)
{
    unsigned int actual_len = 0;
    unsigned int i;

    int16_t sweep_min, sweep_max, sweep_val;

    /* LMS6002D RX DC calibrations have a limited range. libbladeRF throws away
     * the lower 5 bits. */
    const int16_t sweep_inc = 32;

    const int16_t min_est = (i_est < q_est) ? i_est : q_est;
    const int16_t max_est = (i_est > q_est) ? i_est : q_est;

	printf("init_rx_cal_sweep\r\n");
    sweep_min = min_est - 12 * 32;
    if (sweep_min < -2048) {
        sweep_min = -2048;
    }

    sweep_max = max_est + 12 * 32;
    if (sweep_max > 2048) {
        sweep_max = 2048;
    }

    /* Given that these lower bits are thrown away, it can be confusing to
     * see that values change in their LSBs that don't matter. Therefore,
     * we'll adjust to muliples of sweep_inc */
    sweep_min = (sweep_min / 32) * 32;
    sweep_max = (sweep_max / 32) * 32;


    printf("Sweeping [%d : %d : %d]\r\n", sweep_min, sweep_inc, sweep_max);

    sweep_val = sweep_min;
    for (i = 0; sweep_val < sweep_max && i < RX_CAL_MAX_SWEEP_LEN; i++) {
        corr[i] = sweep_val;
        sweep_val += sweep_inc;
        actual_len++;
    }

    *sweep_len = actual_len;
}

static int rx_cal_sweep(struct rx_cal *cal,
                        int16_t *corr, unsigned int sweep_len,
                        int16_t *result_i, int16_t *result_q,
                        float *error_i,  float *error_q)
{
    int status = BLADERF_ERR_UNEXPECTED;
    unsigned int n;

    int16_t min_corr_i = 0;
    int16_t min_corr_q = 0;

    float mean_i, mean_q;
    float min_val_i, min_val_q;

	printf("rx_cal_sweep\r\n");
    min_val_i = min_val_q = 2048;

    for (n = 0; n < sweep_len; n++) {
        status = set_rx_dc_corr(cal->dev, corr[n], corr[n]);
        if (status != 0) {
            return status;
        }

        status = rx_samples(cal->dev, cal->samples, cal->num_samples,
                            &cal->ts, RX_CAL_TS_INC);
        if (status != 0) {
            return status;
        }

        sample_mean(cal->samples, cal->num_samples, &mean_i, &mean_q);

        printf("  Corr=%4d, Mean_I=%4.2f, Mean_Q=%4.2f\r\n",
                   corr[n], mean_i, mean_q);

        /* Not using fabs() to avoid adding a -lm dependency */
        if (mean_i < 0) {
            mean_i = -mean_i;
        }

        if (mean_q < 0) {
            mean_q = -mean_q;
        }

        if (mean_i < min_val_i) {
            min_val_i  = mean_i;
            min_corr_i = corr[n];
        }

        if (mean_q < min_val_q) {
            min_val_q  = mean_q;
            min_corr_q = corr[n];
        }
    }

    *result_i = min_corr_i;
    *result_q = min_corr_q;
    *error_i  = min_val_i;
    *error_q  = min_val_q;

    return 0;
}

int get_dc(volatile int16_t *p_adc, int imag)
{
	int					i;
	int					offset	= (imag != 0) ? 1 : 0;
	int					dc		= 0;
	volatile int16_t	*p_adc2;

	p_adc2 = p_adc + offset;

	for (i = 0; i < 4096; i += 2)	// 2048 samples; EURECOM=128
	{
		dc += p_adc2[i];
	}
	(dc >>= 11);	// /2048 samples; EURECOM=6
	return (dc);
}

static int perform_rx_cal(struct rx_cal *cal, struct dc_calibration_params *p)
{
    int				status	= 0;
    int16_t			i_est;
    int16_t			q_est;
    unsigned int	sweep_len = RX_CAL_MAX_SWEEP_LEN;	// 128

	int				stepsize	= 64;
	int				dc;
	int				dc_off_min	= 99999;
	int				rxdc		= 0;
	int				rf_rxdc;
	unsigned int	rxdc_s;
	unsigned int	rxdc_min	= 31;

	int16_t			*p_adc		= NULL;

	printf("perform_rx_cal\r\n");

	status = rx_cal_update_frequency(cal, p->frequency);
    if (status != 0) {
        return status;
    }

#if 1	// EURECOM implementation
	// Enable only RX for frequency setting
//	LMS_WRITE(cal->dev, 0x05, 0x32 | (1<<2));	// enable Top/TX/RX software
//	tdd_fdd(0/*FDD*/, 0, 0);					//
//	LMS_WRITE(cal->dev, 0x0A, 0x00);	// choose FDD mode
//	LMS_WRITE(cal->dev, 0x08, 0x00);			// LBRFEN: loopback disabled

//	lms_read_registers(cal->dev);
//	lms_read_cal_registers(cal->dev);

	// RX LO Leakage Calibration
#ifdef DEBUG_LMS6002
	printf("lime %d: RX DC tune\n", ID_6002);
#endif

	get_rx_samples(0);
	p_adc	= (uint16_t *) openair0_exmimo_pci[0].adc_head[0];

	while (stepsize >= 1)
	{
		stepsize = stepsize >> 1;

		if (rxdc < 0)
			rxdc_s = 64 + abs(rxdc);
		else
			rxdc_s = rxdc;

		rf_rxdc		= (128 + rxdc_s);

		LMS_WRITE(cal->dev, 0x71, 0x80 | ((rf_rxdc & 0x7f)));	// DC I path cancellation

//		delay_ms(1);
		usleep(1000);
		get_rx_samples(0);
		dc = get_dc(p_adc, 0);
//#ifdef DEBUG_LMS6002
		printf("lime %d: RX DC I : dc=%d (rf_rxdc=%d)\r\n",0,dc,rf_rxdc);
//#endif
		if (abs(dc) < dc_off_min)
		{
			dc_off_min	= abs(dc);
			rxdc_min	= rf_rxdc;
			printf("lime %d: RX DC I : rxdc_min=0x%.2x\r\n", 0, rxdc_min&0x7f);
		}

		if (dc > 0)
			rxdc	= rxdc + stepsize;
		else
			rxdc	= rxdc - stepsize;
	}
	LMS_WRITE(cal->dev, 0x71, 0x80 | ((rxdc_min&0x7f)));	// DC I path cancellation
	printf("lime %d: RX DC I : %d (%d)\r\n", 0, dc_off_min, (rf_rxdc - 128) );
	printf("lime %d: RX DC I Final : rxdc_min=0x%.2x\r\n", 0, rxdc_min&0x7f);

	stepsize	= 64;
	rxdc_s		= 0;
	rxdc		= 0;

	dc_off_min	= 99999;
	rxdc_min	= 31;

	while (stepsize >= 1)
	{
		stepsize	= stepsize >> 1;

		if (rxdc < 0)
			rxdc_s = 64 + abs(rxdc);
		else
			rxdc_s = rxdc;

		rf_rxdc   = (128+rxdc_s);

		LMS_WRITE(cal->dev, 0x72, 0x80 | ((rf_rxdc&0x7f)));	// DC Q path cancellation

//		delay_ms(1);
		usleep(1000);
		get_rx_samples(0);
		dc	= get_dc(p_adc, 1);
//#ifdef DEBUG_LMS6002
		printf("lime %d: RX DC Q : dc=%d (rf_rxdc=%d)\r\n", 0, dc, rf_rxdc);
//#endif
		if (abs(dc) < dc_off_min)
		{
			dc_off_min	= abs(dc);
			rxdc_min	= rf_rxdc;
			printf("lime %d: RX DC Q : rxdc_min=0x%.2x\r\n", 0, rxdc_min&0x7f);
		}

		if (dc > 0)
			rxdc	= rxdc + stepsize;
		else
			rxdc	= rxdc - stepsize;
	}
	LMS_WRITE(cal->dev, 0x72, 0x80 | ((rxdc_min&0x7f)));	// DC Q path cancellation
	printf("lime %d: RX DC Q : %d (%d)\r\n", 0, dc_off_min, rf_rxdc-128);
	printf("lime %d: RX DC Q Final : rxdc_min=0x%.2x\r\n", 0, rxdc_min&0x7f);

//	stepsize	= 64;
//	rxdc_s		= 0;
//	rxdc		= 0;
//
//	dc_off_min	= 99999;
//	rxdc_min	= 31;
//
//	while (stepsize >= 1)
//	{
//		stepsize	= stepsize >> 1;
//
//		if (rxdc < 0)
//			rxdc_s	= 64 + abs(rxdc);
//		else
//			rxdc_s	= rxdc;
//
//		rf_rxdc   = (128 + rxdc_s);
//
//		delay_ms(1);
//		dc = get_dc(p_adc, 0);
//#ifdef DEBUG_LMS6002
//		printf("lime %d: RX DC Q : %d (%d)\n", ID_6002, dc, rf_rxdc);
//#endif
//		if (abs(dc) < dc_off_min)
//		{
//			dc_off_min	= abs(dc);
//			rxdc_min	= rf_rxdc;
//		}
//
//		if (dc > 0)
//			rxdc	= rxdc + stepsize;
//		else
//			rxdc	= rxdc - stepsize;
//	}
    status	= 0;
#else	// BLADERF implementation
    /* Get an initial guess at our correction values */
    status = rx_cal_coarse_estimate(cal, &i_est, &q_est);
    if (status != 0) {
        return status;
    }

    /* Perform a finer sweep of correction values */
    init_rx_cal_sweep(cal->corr_sweep, &sweep_len, i_est, q_est);

    /* Advance our timestmap just to account for any time we may have lost */
    cal->ts += RX_CAL_TS_INC;

    status = rx_cal_sweep(cal, cal->corr_sweep, sweep_len,
                          &p->corr_i, &p->corr_q,
                          &p->error_i, &p->error_q);
#endif
    return status;
}

static int rx_cal_init_state(struct bladerf *dev,
                             const struct rx_cal_backup *backup,
                             struct rx_cal *state)
{
    int		status		= 0;
	short	*dump_sig	= NULL;
	printf("rx_cal_init_state\r\n");

	state->dev = dev;
	state->num_samples  = 76800;	// 5 * 3000000 / 1000 = 15000
//    state->num_samples  = RX_CAL_COUNT;	// 5 * 3000000 / 1000 = 15000
//
//    state->samples = malloc(2 * sizeof(state->samples[0]) * RX_CAL_COUNT);	// 15000
//    if (state->samples == NULL) {
//        return BLADERF_ERR_MEM;
//    }
	state->samples	= (uint16_t *) openair0_exmimo_pci[0].adc_head[0];
	dump_sig = (short*) openair0_exmimo_pci[0].adc_head[0];
	printf("dump_sig = 0x%08lx\r\n", (long unsigned int)dump_sig);
//	for( j=0; j<128;j++ )	
//	{
//		printf("i=%d rx_sig[I]=0x%hx, rx_sig[Q]=0x%hx, Ox%lx\n", j, dump_sig[j*2], dump_sig[j*2+1], (unsigned long)dump_sig[j*2]);
//	}

    state->corr_sweep = malloc(sizeof(state->corr_sweep[0]) * RX_CAL_MAX_SWEEP_LEN);	// 128
    if (state->corr_sweep == NULL) {
        return BLADERF_ERR_MEM;
    }

    state->tx_freq = backup->tx_freq;

//    status = bladerf_get_timestamp(dev, BLADERF_MODULE_RX, &state->ts);
//    if (status != 0) {
//        return status;
//    }

    /* Schedule first RX well into the future */
//    state->ts += 20 * RX_CAL_TS_INC;	// 15 * 3000000 / 1000 = 45000

    return status;
}

static int rx_cal_init(struct bladerf *dev)
{
    int status;

	printf("rx_cal_init\r\n");

//    status = bladerf_set_sample_rate(dev, BLADERF_MODULE_RX, RX_CAL_RATE, NULL);
//    if (status != 0) {
//        return status;
//    }

    status = bladerf_set_bandwidth(dev, BLADERF_MODULE_RX, RX_CAL_BW, NULL);
    if (status != 0) {
        return status;
    }

//    status = bladerf_sync_config(dev, BLADERF_MODULE_RX,
//                                 BLADERF_FORMAT_SC16_Q11_META,
//                                 64, 16384, 16, 1000);
//    if (status != 0) {
//        return status;
//    }

    status = bladerf_enable_module(dev, BLADERF_MODULE_RX, true);
    if (status != 0) {
        return status;
    }

    return status;
}

static int set_rx_cal_backup(struct bladerf *dev, struct rx_cal_backup *b)
{
    int status;
    int retval = 0;

//    status = bladerf_set_rational_sample_rate(dev, BLADERF_MODULE_RX,
//                                              &b->rational_sample_rate, NULL);
//    if (status != 0 && retval == 0) {
//        retval = status;
//    }

    status = bladerf_set_bandwidth(dev, BLADERF_MODULE_RX, b->bandwidth, NULL);
    if (status != 0 && retval == 0) {
        retval = status;
    }

    status = bladerf_set_frequency(dev, BLADERF_MODULE_TX, b->tx_freq);
    if (status != 0 && retval == 0) {
        retval = status;
    }

    return retval;
}

int dc_calibration_rx(void *dev,
                      struct dc_calibration_params *params,
                      size_t params_count, bool print_status)
{
    int status = 0;
    int retval = 0;
    struct rx_cal state;
    struct rx_cal_backup backup;
    size_t i;

	printf("dc_calibration_rx\r\n");

    memset(&state, 0, sizeof(state));

    status = get_rx_cal_backup(dev, &backup);
    if (status != 0) {
        return status;
    }

    status = rx_cal_init(dev);
    if (status != 0) {
        goto out;
    }

    status = rx_cal_init_state(dev, &backup, &state);
    if (status != 0) {
        goto out;
    }

    for (i = 0; i < params_count && status == 0; i++) {
        status = perform_rx_cal(&state, &params[i]);

        if (status == 0 && print_status) {
#           ifdef DEBUG_DC_CALIBRATION
            const char sol = '\n';
            const char eol = '\n';
#           else
            const char sol = '\r';
            const char eol = '\0';
#           endif
            printf("%cCalibrated @ %10u Hz: I=%4d (Error: %4.2f), "
                   "Q=%4d (Error: %4.2f)      %c",
                   sol,
                   params[i].frequency,
                   params[i].corr_i, params[i].error_i,
                   params[i].corr_q, params[i].error_q,
                   eol);
            fflush(stdout);
        }
    }

    if (print_status) {
        putchar('\n');
    }

out:
//    free(state.samples);
    free(state.corr_sweep);

    retval = status;

    status = bladerf_enable_module(dev, BLADERF_MODULE_RX, false);
    if (status != 0 && retval == 0) {
        retval = status;
    }

    status = set_rx_cal_backup(dev, &backup);
    if (status != 0 && retval == 0) {
        retval = status;
    }

    return retval;
}

int dc_calibration(void				*dev,
					bladerf_module	module,
					struct dc_calibration_params	*params,
					size_t			num_params,
					bool			show_status)
{
	int status;

	printf("dc_calibration\r\n");

	cal_dc_init(dev);

	switch (module)
	{
		case BLADERF_MODULE_RX:
			status = dc_calibration_rx(dev, params, num_params, show_status);
			break;
		case BLADERF_MODULE_TX:
			status = dc_calibration_tx(dev, params, num_params, show_status);
			break;
		default:
			status = BLADERF_ERR_INVAL;
	}

	cal_dc_deinit(dev);

	return status;
}
