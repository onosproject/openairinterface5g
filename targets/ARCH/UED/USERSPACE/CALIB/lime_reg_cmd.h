/*
 * lime_reg_cmd.h
 *
 *  Created on: 5 févr. 2016
 *      Author: root
 */

#ifndef SYRPCIEAPP_ARCH_EXMIMO_USERSPACE_CALIB_LIME_REG_CMD_H_
#define SYRPCIEAPP_ARCH_EXMIMO_USERSPACE_CALIB_LIME_REG_CMD_H_

void lms_read_registers(void *cmdcontext_spi);
void lms_read_cal_registers(void *cmdcontext_spi);

#endif /* SYRPCIEAPP_ARCH_EXMIMO_USERSPACE_CALIB_LIME_REG_CMD_H_ */
