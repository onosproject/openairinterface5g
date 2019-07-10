#!/bin/bash

sudo iptables -t mangle -F
sudo iptables -t nat -F

# Load ue_ip module
sudo rmmod ue_ip
source oaienv
cd cmake_targets/tools
source init_nas_s1 UE

#Interface configuration
sudo ifconfig oip0 10.0.0.2

sudo ifconfig oip0 hw ether 00:00:00:00:00:02

#Routing configuration for specific sidelink destination addresses (sidelink (10.0.0.1))
sudo ip neigh add 10.0.0.1 lladdr 00:00:00:00:00:01 dev oip0 nud permanent
sudo ip neigh change 10.0.0.1 lladdr 00:00:00:00:00:01 dev oip0 nud permanent

# Mark outgoing UNICAST traffic to UE 10.0.0.2 with the SLRB ID
sudo iptables -A POSTROUTING  -t mangle -o oip0 -d 10.0.0.1 -j MARK --set-mark 4 

# Mark outgoing MULTICAST traffic to group 224.0.0.3 with the SLRB ID
sudo iptables -A POSTROUTING  -t mangle -o oip0 -d 224.0.0.3 -j MARK --set-mark 5

# Associate destination address (e.g. 8.8.8.8) with the MAC address of the relay UE at the neighbor table
sudo ip neigh add 8.8.8.8 lladdr 00:00:00:00:00:01 dev oip0 nud permanent

# Mark outgoing packets for specific destination address (e.g. 8.8.8.8) with SLRB id
sudo iptables -A POSTROUTING  -t mangle -o oip0 -d 8.8.8.8 -j MARK --set-mark 4
