#!/bin/bash

#Interface configuration
sudo ifconfig oip0 10.0.0.2

sudo ifconfig oip0 hw ether 00:00:00:00:00:02

#Routing configuration for specific sidelink destination addresses (sidelink (10.0.0.1))
sudo ip neigh add 10.0.0.1 lladdr 00:00:00:00:00:01 dev oip0 nud permanent

#Establishment of SLRB for sidelink communication
sudo iptables -A POSTROUTING  -t mangle -o oip0 -d 10.0.0.1 -j MARK --set-mark 4 

# Associate destination address (e.g. 8.8.8.8) with the MAC address of the relay UE at the neighbor table
sudo ip neigh add 8.8.8.8 lladdr 00:00:00:00:00:01 dev oip0 nud permanent

# Mark outgoing packets for specific destination address (e.g. 8.8.8.8) with SLRB id
sudo iptables -A POSTROUTING  -t mangle -o oip0 -d 8.8.8.8 -j MARK --set-mark 4
