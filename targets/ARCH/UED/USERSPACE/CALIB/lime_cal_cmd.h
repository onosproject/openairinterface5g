/*
 * lime_cal_cmd.h
 *
 *  Created on: 5 févr. 2016
 *      Author: root
 */

#ifndef SYRPCIEAPP_ARCH_EXMIMO_USERSPACE_CALIB_LIME_CAL_CMD_H_
#define SYRPCIEAPP_ARCH_EXMIMO_USERSPACE_CALIB_LIME_CAL_CMD_H_

/* Currently, only pthreads is supported. In the future, native windows threads
 * may be used; one of the objectives of this file is to ease that transistion.
 */
#include <stdbool.h>
#include <pthread.h>
#include "lime_spi_cmd.h"

/**
 * If this bit is set, configure PLL output buffers for operation in the
 * bladeRF's "low band." Otherwise, configure the device for operation in the
 * "high band."
 */
#define LMS_FREQ_FLAGS_LOW_BAND     (1 << 0)

/**
 * Use VCOCAP value as-is, rather as using it as a starting point hint
 * to the tuning algorithm.  This offers a faster retune, with a potential
 * trade-off in phase noise.
 */
#define LMS_FREQ_FLAGS_FORCE_VCOCAP   (1 << 1)

/**
 * Information about the frequency calculation for the LMS6002D PLL
 * Calculation taken from the LMS6002D Programming and Calibration Guide
 * version 1.1r1.
 */
struct lms_freq {
    uint8_t     freqsel;    /**< Choice of VCO and dision ratio */
    uint8_t     vcocap;     /**< VCOCAP hint */
    uint16_t    nint;       /**< Integer portion of f_LO given f_REF */
    uint32_t    nfrac;      /**< Fractional portion of f_LO given nint and f_REF */
    uint8_t     flags;      /**< Additional parameters defining the tuning
                                 configuration. See LMFS_FREQ_FLAGS_* values */

#ifndef BLADERF_NIOS_BUILD
    uint8_t     x;         /**< VCO division ratio */
#endif

    uint8_t     vcocap_result;  /**< Filled in by retune operation to denote
                                     which VCOCAP value was used */
};

/**
 * Module selection for those which have both RX and TX constituents
 */
typedef enum
{
    BLADERF_MODULE_INVALID = -1,    /**< Invalid module entry */
    BLADERF_MODULE_RX,              /**< Receive Module */
    BLADERF_MODULE_TX               /**< Transmit Module */
} bladerf_module;

/**
 * Loopback options
 */
typedef enum {

    /**
     * Firmware loopback inside of the FX3
     */
    BLADERF_LB_FIRMWARE = 1,

    /**
     * Baseband loopback. TXLPF output is connected to the RXVGA2 input.
     */
    BLADERF_LB_BB_TXLPF_RXVGA2,

    /**
     * Baseband loopback. TXVGA1 output is connected to the RXVGA2 input.
     */
    BLADERF_LB_BB_TXVGA1_RXVGA2,

    /**
     * Baseband loopback. TXLPF output is connected to the RXLPF input.
     */
    BLADERF_LB_BB_TXLPF_RXLPF,

    /**
     * Baseband loopback. TXVGA1 output is connected to RXLPF input.
     */
    BLADERF_LB_BB_TXVGA1_RXLPF,

    /**
     * RF loopback. The TXMIX output, through the AUX PA, is connected to the
     * output of LNA1.
     */
    BLADERF_LB_RF_LNA1,


    /**
     * RF loopback. The TXMIX output, through the AUX PA, is connected to the
     * output of LNA2.
     */
    BLADERF_LB_RF_LNA2,

    /**
     * RF loopback. The TXMIX output, through the AUX PA, is connected to the
     * output of LNA3.
     */
    BLADERF_LB_RF_LNA3,

    /**
     * Disables loopback and returns to normal operation.
     */
    BLADERF_LB_NONE

} bladerf_loopback;

/**
 * LNA options
 */
typedef enum {
    LNA_NONE,   /**< Disable all LNAs */
    LNA_1,      /**< Enable LNA1 (300MHz - 2.8GHz) */
    LNA_2,      /**< Enable LNA2 (1.5GHz - 3.8GHz) */
    LNA_3       /**< Enable LNA3 (Unused on the bladeRF) */
} lms_lna;

/**
 * PA Selection
 */
typedef enum {
    PA_AUX,         /**< AUX PA Enable (for RF Loopback) */
    PA_1,           /**< PA1 Enable (300MHz - 2.8GHz) */
    PA_2,           /**< PA2 Enable (1.5GHz - 3.8GHz) */
    PA_NONE,        /**< All PAs disabled */
} lms_pa;

/**
 * DC Calibration Modules
 */
typedef enum {
    BLADERF_DC_CAL_INVALID = -1,
    BLADERF_DC_CAL_LPF_TUNING,
    BLADERF_DC_CAL_TX_LPF,
    BLADERF_DC_CAL_RX_LPF,
    BLADERF_DC_CAL_RXVGA2
} bladerf_cal_module;

/**
 * LNA gain options
 */
typedef enum {
    BLADERF_LNA_GAIN_UNKNOWN,    /**< Invalid LNA gain */
    BLADERF_LNA_GAIN_BYPASS,     /**< LNA bypassed - 0dB gain */
    BLADERF_LNA_GAIN_MID,        /**< LNA Mid Gain (MAX-6dB) */
    BLADERF_LNA_GAIN_MAX         /**< LNA Max Gain */
} bladerf_lna_gain;

/**
 * Sample format
 */
typedef enum {
    /**
     * Signed, Complex 16-bit Q11. This is the native format of the DAC data.
     *
     * Values in the range [-2048, 2048) are used to represent [-1.0, 1.0).
     * Note that the lower bound here is inclusive, and the upper bound is
     * exclusive. Ensure that provided samples stay within [-2048, 2047].
     *
     * Samples consist of interleaved IQ value pairs, with I being the first
     * value in the pair. Each value in the pair is a right-aligned,
     * little-endian int16_t. The FPGA ensures that these values are
     * sign-extended.
     *
     * When using this format the minimum required buffer size, in bytes, is:
     * <pre>
     *   buffer_size_min = [ 2 * num_samples * sizeof(int16_t) ]
     * </pre>
     *
     * For example, to hold 2048 samples, a buffer must be at least 8192 bytes
     * large.
     */
    BLADERF_FORMAT_SC16_Q11,

    /**
     * This format is the same as the ::BLADERF_FORMAT_SC16_Q11 format, except the
     * first 4 samples (16 bytes) in every block of 1024 samples are replaced
     * with metadata, organized as follows, with all fields being little endian
     * byte order:
     *
     * <pre>
     *  0x00 [uint32_t:  Reserved]
     *  0x04 [uint64_t:  64-bit Timestamp]
     *  0x0c [uint32_t:  BLADERF_META_FLAG_* flags]
     * </pre>
     *
     * When using the bladerf_sync_rx() and bladerf_sync_tx() functions,
     * this detail is transparent to caller. These functions take care of
     * packing/unpacking the metadata into/from the data, via the
     * bladerf_metadata structure.
     *
     * Currently, when using the asynchronous data transfer interface, the user
     * is responsible for manually packing/unpacking this metadata into/from
     * their sample data.
     */
    BLADERF_FORMAT_SC16_Q11_META,
} bladerf_format;

/*
 * Metadata flags
 *
 * These are used in conjunction with the bladerf_metadata structure's
 * `flags` field.
 */

/**
 * Mark the associated buffer as the start of a burst transmission.
 * This is only used for the bladerf_sync_tx() call.
 *
 * When using this flag, the bladerf_metadata::timestamp field should contain
 * the timestamp at which samples should be sent.
 *
 * Between specifying the ::BLADERF_META_FLAG_TX_BURST_START and
 * ::BLADERF_META_FLAG_TX_BURST_END flags, there is no need for the user to the
 * bladerf_metadata::timestamp field because the library will ensure the
 * correct value is used, based upon the timestamp initially provided and
 * the number of samples that have been sent.
 */
#define BLADERF_META_FLAG_TX_BURST_START   (1 << 0)

/**
 * Mark the associated buffer as the end of a burst transmission. This will
 * flush the remainder of the sync interface's current working buffer and
 * enqueue samples into the hardware's transmit FIFO.
 *
 * As of libbladeRF v1.3.0, it is no longer necessary for the API user to
 * ensure that the final 3 samples of a burst are 0+0j. libbladeRF now ensures
 * this hardware requirement (driven by the LMS6002D's pre-DAC register stages)
 * is upheld.
 *
 * Specifying this flag and flushing the sync interface's working buffer implies
 * that the next timestamp that can be transmitted is the current timestamp plus
 * the duration of the burst that this flag is ending <b>and</b> the remaining
 * length of the remaining buffer that is flushed. (The buffer size, in this
 * case, is the `buffer_size` value passed to the previous bladerf_sync_config()
 * call.)
 *
 * Rather than attempting to keep track of the number of samples sent with
 * respect to buffer sizes, it is easiest to always assume 1 buffer's worth of
 * time is required between bursts. In this case "buffer" refers to the
 * `buffer_size` parameter provided to bladerf_sync_config().) If this is too
 * much time, consider using the ::BLADERF_META_FLAG_TX_UPDATE_TIMESTAMP
 * flag.
 *
 * This is only used for the bladerf_sync_tx() call. It is ignored by the
 * bladerf_sync_rx() call.
 *
 */
#define BLADERF_META_FLAG_TX_BURST_END     (1 << 1)

/**
 * Use this flag in conjunction with ::BLADERF_META_FLAG_TX_BURST_START to
 * indicate that the burst should be transmitted as soon as possible, as opposed
 * to waiting for a specific timestamp.
 *
 * When this flag is used, there is no need to set the
 * bladerf_metadata::timestamp field.
 */
#define BLADERF_META_FLAG_TX_NOW           (1 << 2)

/**
 * Use this flag within a burst (i.e., between the use of
 * ::BLADERF_META_FLAG_TX_BURST_START and ::BLADERF_META_FLAG_TX_BURST_END) to
 * specify that bladerf_sync_tx() should read the bladerf_metadata::timestamp
 * field and zero-pad samples up to the specified timestamp. The provided
 * samples will then be transmitted at that timestamp.
 *
 * Use this flag when potentially flushing an entire buffer via the
 * ::BLADERF_META_FLAG_TX_BURST_END would yield an unacceptably large gap in
 * the transmitted samples.
 *
 * In some applications where a transmitter is constantly transmitting
 * with extremely small gaps (less than a buffer), users may end up using a
 * single ::BLADERF_META_FLAG_TX_BURST_START, and then numerous calls to
 * bladerf_sync_tx() with the ::BLADERF_META_FLAG_TX_UPDATE_TIMESTAMP
 * flag set.  The ::BLADERF_META_FLAG_TX_BURST_END would only be used to end
 * the stream when shutting down.
 */
#define BLADERF_META_FLAG_TX_UPDATE_TIMESTAMP (1 << 3)

/**
 * This flag indicates that calls to bladerf_sync_rx should return any available
 * samples, rather than wait until the timestamp indicated in the
 * bladerf_metadata timestamp field.
 */
#define BLADERF_META_FLAG_RX_NOW           (1 << 31)

/**
 * Sample metadata
 *
 * This structure is used in conjunction with the ::BLADERF_FORMAT_SC16_Q11_META
 * format to TX scheduled bursts or retrieve timestamp information about
 * received samples.
 */
struct bladerf_metadata {

    /**
     * Free-running FPGA counter that monotonically increases at the
     * sample rate of the associated module. */
    uint64_t timestamp;

    /**
     * Input bit field to control the behavior of the call that the metadata
     * structure is passed to. API calls read this field from the provided
     * data structure, and do not modify it.
     *
     * Valid flags include
     *  ::BLADERF_META_FLAG_TX_BURST_START, ::BLADERF_META_FLAG_TX_BURST_END,
     *  ::BLADERF_META_FLAG_TX_NOW, ::BLADERF_META_FLAG_TX_UPDATE_TIMESTAMP,
     *  and ::BLADERF_META_FLAG_RX_NOW
     *
     */
    uint32_t flags;

    /**
     * Output bit field to denoting the status of transmissions/receptions. API
     * calls will write this field.
     *
     * Possible status flags include ::BLADERF_META_STATUS_OVERRUN and
     * ::BLADERF_META_STATUS_UNDERRUN;
     *
     */
    uint32_t status;

    /**
     * This output parameter is updated to reflect the actual number of
     * contiguous samples that have been populated in an RX buffer during
     * a bladerf_sync_rx() call.
     *
     * This will not be equal to the requested count in the event of a
     * discontinuity (i.e., when the status field has the
     * ::BLADERF_META_STATUS_OVERRUN flag set). When an overrun occurs, it is
     * important not to read past the number of samples specified by this
     * value, as the remaining contents of the buffer are undefined.
     *
     * This parameter is not currently used by bladerf_sync_tx().
     */
    unsigned int actual_count;

    /**
     * Reserved for future use. This is not used by any functions.
     * It is recommended that users zero out this field.
     */
    uint8_t reserved[32];
};

/**
 * Rational sample rate representation
 */
struct bladerf_rational_rate {
    uint64_t integer;           /**< Integer portion */
    uint64_t num;               /**< Numerator in fractional portion */
    uint64_t den;               /**< Denominator in fractional portion. This
                                     must be > 0. */
};

/**
 * LPF mode
 */
typedef enum {
    BLADERF_LPF_NORMAL,     /**< LPF connected and enabled */
    BLADERF_LPF_BYPASSED,   /**< LPF bypassed */
    BLADERF_LPF_DISABLED    /**< LPF disabled */
} bladerf_lpf_mode;

/**
 * Set the frequency of a module in Hz
 *
 * @param[in]   dev     Device handle
 * @param[in]   mod     Module to change
 * @param[in]   freq    Frequency in Hz to tune
 *
 * @return 0 on success, BLADERF_ERR_* value on failure
 */
static inline int lms_set_frequency(void *dev,
                                    bladerf_module mod, uint32_t freq)
{
    struct lms_freq f;
    lms_calculate_tuning_params(freq, &f);
    return lms_set_precalculated_frequency(dev, mod, &f);
}

/**
 * Wrapper for setting bits in an LMS6002 register via a RMW operation
 *
 * @param   dev         Device to operate on
 * @param   addr        Register address
 * @param   mask        Bits to set should be '1'
 *
 * @return BLADERF_ERR_* value
 */
static inline int lms_set(void *dev, uint8_t addr, uint8_t mask)
{
	int		status	= 0;
	uint8_t	regval	= 0;

	status = LMS_READ(dev, addr, &regval);
	if (status != 0)
	{
		return status;
	}

	regval |= mask;

	return LMS_WRITE(dev, addr, regval);
}

/*
 * Wrapper for clearing bits in an LMS6002 register via a RMW operation
 *
 * @param   dev         Device to operate on
 * @param   addr        Register address
 * @param   mask        Bits to clear should be '1'
 *
 * @return BLADERF_ERR_* value
 */
static inline int lms_clear(void *dev, uint8_t addr, uint8_t mask)
{
	int		status	= 0;
	uint8_t	regval	= 0;

	status = LMS_READ(dev, addr, &regval);
	if (status != 0)
	{
		return status;
	}

	regval &= ~mask;

	return LMS_WRITE(dev, addr, regval);
}



struct dc_calibration_params {
    unsigned int frequency;
    int16_t corr_i;
    int16_t corr_q;
    float error_i;
    float error_q;
};

/**
 * Internal low-pass filter bandwidth selection
 */
typedef enum {
    BW_28MHz,       /**< 28MHz bandwidth, 14MHz LPF */
    BW_20MHz,       /**< 20MHz bandwidth, 10MHz LPF */
    BW_14MHz,       /**< 14MHz bandwidth, 7MHz LPF */
    BW_12MHz,       /**< 12MHz bandwidth, 6MHz LPF */
    BW_10MHz,       /**< 10MHz bandwidth, 5MHz LPF */
    BW_8p75MHz,     /**< 8.75MHz bandwidth, 4.375MHz LPF */
    BW_7MHz,        /**< 7MHz bandwidth, 3.5MHz LPF */
    BW_6MHz,        /**< 6MHz bandwidth, 3MHz LPF */
    BW_5p5MHz,      /**< 5.5MHz bandwidth, 2.75MHz LPF */
    BW_5MHz,        /**< 5MHz bandwidth, 2.5MHz LPF */
    BW_3p84MHz,     /**< 3.84MHz bandwidth, 1.92MHz LPF */
    BW_3MHz,        /**< 3MHz bandwidth, 1.5MHz LPF */
    BW_2p75MHz,     /**< 2.75MHz bandwidth, 1.375MHz LPF */
    BW_2p5MHz,      /**< 2.5MHz bandwidth, 1.25MHz LPF */
    BW_1p75MHz,     /**< 1.75MHz bandwidth, 0.875MHz LPF */
    BW_1p5MHz,      /**< 1.5MHz bandwidth, 0.75MHz LPF */
} lms_bw;

/**
 * Correction parameter selection
 *
 * These values specify the correction parameter to modify or query when
 * calling bladerf_set_correction() or bladerf_get_correction(). Note that the
 * meaning of the `value` parameter to these functions depends upon the
 * correction parameter.
 *
 */
typedef enum {
    /**
     * Adjusts the in-phase DC offset via controls provided by the LMS6002D
     * front end. Valid values are [-2048, 2048], which are scaled to the
     * available control bits in the LMS device.
     */
    BLADERF_CORR_LMS_DCOFF_I,

    /**
     * Adjusts the quadrature DC offset via controls provided the LMS6002D
     * front end. Valid values are [-2048, 2048], which are scaled to the
     * available control bits.
     */
    BLADERF_CORR_LMS_DCOFF_Q,

    /**
     * Adjusts FPGA-based phase correction of [-10, 10] degrees, via a provided
     * count value of [-4096, 4096].
     */
    BLADERF_CORR_FPGA_PHASE,

    /**
     * Adjusts FPGA-based gain correction value in [-1.0, 1.0], via provided
     * values in the range of [-4096, 4096].
     */
    BLADERF_CORR_FPGA_GAIN
} bladerf_correction;

/*
 * Metadata status bits
 *
 * These are used in conjunction with the bladerf_metadata structure's
 * `status` field.
 */

/**
 * A sample overrun has occurred. This indicates that either the host
 * (more likely) or the FPGA is not keeping up with the incoming samples
 */
#define BLADERF_META_STATUS_OVERRUN  (1 << 0)

/**
 * A sample underrun has occurred. This generally only occurrs on the TX module
 * when the FPGA is starved of samples.
 *
 * @note libbladeRF does not report this status. It is here for future use.
 */
#define BLADERF_META_STATUS_UNDERRUN (1 << 1)



unsigned char	tunevcocap_tx(void *cmdcontext_spi);
unsigned char	tunevcocap_rx(void *cmdcontext_spi);

#endif /* SYRPCIEAPP_ARCH_EXMIMO_USERSPACE_CALIB_LIME_CAL_CMD_H_ */
