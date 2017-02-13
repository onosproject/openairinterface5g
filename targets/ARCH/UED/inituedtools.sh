#!/bin/bash

# ********* ********* ********* ********* ********* ********* ********* *****
# ex do_reenumerate_expressmimo.sh

# Re-Enumerate FPGA card

# After resetting or powering up the card or after reloading the FPGA bitstream,
# run this script to re-enumerate the PCIe device in Linux.

# You may need to change the device path. Check lspci output for this.

# You need to run this as root:
# sudo ./init_syrpcietools.sh

# Matthias <ihmig@eurecom.fr>, 2013

rmmod openair_rf
echo 1 > /sys/bus/pci/devices/0000\:07\:00.0/remove
echo 1 > /sys/bus/pci/rescan

# ********* ********* ********* ********* ********* ********* ********* *****
# ex init_sdrsyrtem.sh
# export OPENAIR_TARGETS=/home/ued/openair5g/uedtools

PCI=`lspci -m | grep Xilinx`
if [ -z "$PCI" ]; then
 echo "No card found. Stopping!"
 return
fi

## This part corrects the wrong configuration of the endpoint done by the bios in some machines
echo "$PCI" | while read config_reg; do
SLOT_NUMBER=`echo $config_reg | awk -F\" '{print $1}'`
sudo setpci -s $SLOT_NUMBER 60.b=10
done

load_module() {
  mod_name=${1##*/}
  mod_name=${mod_name%.*}
  if awk "/$mod_name/ {found=1 ;exit} END {if (found!=1) exit 1}" /proc/modules
    then
      echo "module $mod_name already loaded: I remove it first"
      sudo rmmod $mod_name
  fi
  echo loading $mod_name
  sudo insmod $1
}

load_module $OPENAIR_TARGETS/bin/updatefw/openair_rf.ko
sleep 1

if [ ! -e /dev/openair0 ]; then 
 sudo mknod /dev/openair0 c 127 0
 sudo chmod a+rw /dev/openair0
fi

DEVICE=`echo $PCI | awk -F\" '{print $(NF-1)}' | awk '{print $2}'`
DEVICE_SWID=${DEVICE:2:2}
if [ $DEVICE_SWID == '08' ]; then	
 echo "Using firmware version 8"
 $OPENAIR_TARGETS/ARCH/UED/USERSPACE/OAI_FW_INIT/updatefw -s 0x43fffff0 -b -f $OPENAIR_TARGETS/ARCH/UED/USERSPACE/OAI_FW_INIT/sdr_expressmimo2
else 
 if [ $DEVICE_SWID == '09' ]; then
  echo "Using firmware version 9"
  $OPENAIR_TARGETS/ARCH/UED/USERSPACE/OAI_FW_INIT/updatefw -s 0x43fffff0 -b -f $OPENAIR_TARGETS/ARCH/UED/USERSPACE/OAI_FW_INIT/sdr_expressmimo2_v9
 else
  if [ $DEVICE_SWID == '0a' ]; then
   echo "Using firware version 10"
   $OPENAIR_TARGETS/bin/updatefw -s 0x43fffff0 -b -f $OPENAIR_TARGETS/ARCH/UED/FPGA/LEON3FW/sdrSyrRf201701301408FMC

#sdrSyrRf201606290937FMC  -- Test with incremental uint32_t counter for rxcnt_ptr[2]
#sdrSyrRf201604221017FMC  -- TDD sans param update + LNA GAIN mode control
#sdrSyrRf201604131017FMC  -- TDD sans param update
#sdrSyrRf201604121624FMC  -- TDD avec param update
#sdrSyrRf201608091835FMC  -- TDD firmware for D3.1 Test Report (08/2016)
#sdrSyrRf201603021208FMC  -- FDD
#sdrSyrRf201608191152FMC  -- FDD with GPIO control LNA2 forced
#sdrSyrRf201608191354FMC  -- FDD with GPIO control LNA1 forced
#sdrSyrRf201608191413FMC  -- FDD with GPIO control depending on frequency (!! last stable !! )
#sdrSyrRf201701091628FMC_UPDATE  -- 12v12 debug update params
#sdrSyrRf201701091627FMC		 -- 12v12 debug no param update
#sdrSyrRf201701101336FMC_SYNCRAM --12v12 syncran bug 
#sdrSyrRf201701101449FMC		 -- sycram timeout extended 128->1024
#sdrSyrRf201701191548FMC -> mesure freq set
#sdrSyrRf201701231604FMC -> POST HARMONY 5MHZ
#sdrSyrRf201701301345FMC -> POSt Harmony 10MHz

   echo 1 > /proc/irq/17/smp_affinity
   cat /proc/irq/17/smp_affinity
  else
   echo 'No corresponding firmware found'
   return
  fi
 fi
fi

