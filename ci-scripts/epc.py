#-----------------------------------------------------------
# Import
#-----------------------------------------------------------
import sys              # arg
import re               # reg
import logging
import os
import time
from multiprocessing import Process, Lock, SimpleQueue

#-----------------------------------------------------------
# OAI Testing modules
#-----------------------------------------------------------
import sshconnection as SSH 
#from sshconnection import SSHConnection as SSH
import helpreadme as HELP
import constants as CONST

#-----------------------------------------------------------
# Class Declaration
#-----------------------------------------------------------
class EPCManagement():
	def __init__(self):
		self.EPCIPAddress = ''
		self.EPCUserName = ''
		self.EPCPassword = ''
		self.EPCSourceCodePath = ''
		self.EPCType = ''
		self.EPC_PcapFileName = ''

	def SetIPAddress(self, ipaddress):
		self.EPCIPAddress = ipaddress
	def GetIPAddress(self):
		return self.EPCIPAddress
	def SetUserName(self, username):
		self.EPCUserName = username
	def GetUserName(self):
		return self.EPCUserName
	def SetPassword(self, password):
		self.EPCPassword = password
	def GetPassword(self):
		return self.EPCPassword
	def SetSourceCodePath(self, sourcecodepath):
		self.EPCSourceCodePath = sourcecodepath
	def GetSourceCodePath(self):
		return self.EPCSourceCodePath
	def SetType(self, typ):
		self.EPCType = typ
	def GetType(self):
		return self.EPCType
	def Set_PcapFileName(self, pcapfn):
		self.PcapFileName = pcapfn
	def Get_PcapFileName(self):
		return self.PcapFileName

	def InitializeHSS(self):
		if self.EPCIPAddress == '' or self.EPCUserName == '' or self.EPCPassword == '' or self.EPCSourceCodePath == '' or self.EPCType == '':
			HELP.GenericHelp(Version)
			HELP.EPCSrvHelp(self.EPCIPAddress, self.EPCUserName, self.EPCPassword, self.EPCSourceCodePath, self.EPCType)
			sys.exit('Insufficient EPC Parameters')
		#mySSH = SSH() 
		mySSH = SSH.SSHConnection() 
		mySSH.open(self.EPCIPAddress, self.EPCUserName, self.EPCPassword)
		if re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
			logging.debug('Using the OAI EPC Release 14 Cassandra-based HSS')
			mySSH.command('cd ' + self.EPCSourceCodePath + '/scripts', '\$', 5)
			logging.debug('\u001B[1m Launching tshark on all interfaces \u001B[0m')
			EPC_PcapFileName = 'epc_' + self.testCase_id + '.pcap'
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S rm -f ' + self.EPC_PcapFileName, '\$', 5)
			mySSH.command('echo $USER; nohup sudo tshark -f "tcp port not 22 and port not 53" -i any -w ' + self.EPCSourceCodePath + '/scripts/' + self.EPC_PcapFileName + ' > /tmp/tshark.log 2>&1 &', self.EPCUserName, 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S mkdir -p logs', '\$', 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S rm -f hss_' + self.testCase_id + '.log logs/hss*.*', '\$', 5)
			mySSH.command('echo "oai_hss -j /usr/local/etc/oai/hss_rel14.json" > ./my-hss.sh', '\$', 5)
			mySSH.command('chmod 755 ./my-hss.sh', '\$', 5)
			mySSH.command('sudo daemon --unsafe --name=hss_daemon --chdir=' + self.EPCSourceCodePath + '/scripts -o ' + self.EPCSourceCodePath + '/scripts/hss_' + self.testCase_id + '.log ./my-hss.sh', '\$', 5)
		elif re.match('OAI', self.EPCType, re.IGNORECASE):
			logging.debug('Using the OAI EPC HSS')
			mySSH.command('cd ' + self.EPCSourceCodePath, '\$', 5)
			mySSH.command('source oaienv', '\$', 5)
			mySSH.command('cd scripts', '\$', 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S ./run_hss 2>&1 | stdbuf -o0 awk \'{ print strftime("[%Y/%m/%d %H:%M:%S] ",systime()) $0 }\' | stdbuf -o0 tee -a hss_' + self.testCase_id + '.log &', 'Core state: 2 -> 3', 35)
		elif re.match('ltebox', self.EPCType, re.IGNORECASE):
			logging.debug('Using the ltebox simulated HSS')
			mySSH.command('if [ -d ' + self.EPCSourceCodePath + '/scripts ]; then echo ' + self.EPCPassword + ' | sudo -S rm -Rf ' + self.EPCSourceCodePath + '/scripts ; fi', '\$', 5)
			mySSH.command('mkdir -p ' + self.EPCSourceCodePath + '/scripts', '\$', 5)
			mySSH.command('cd /opt/hss_sim0609', '\$', 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S rm -f hss.log daemon.log', '\$', 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S echo "Starting sudo session" && sudo daemon --unsafe --name=simulated_hss --chdir=/opt/hss_sim0609 ./starthss_real  ', '\$', 5)
		else:
			logging.error('This option should not occur!')
		mySSH.close()
		#self.CreateHtmlTestRow(self.EPCType, 'OK', CONST.ALL_PROCESSES_OK)

	def InitializeMME(self):
		if self.EPCIPAddress == '' or self.EPCUserName == '' or self.EPCPassword == '' or self.EPCSourceCodePath == '' or self.EPCType == '':
			HELP.GenericHelp(Version)
			HELP.EPCSrvHelp(self.EPCIPAddress, self.EPCUserName, self.EPCPassword, self.EPCSourceCodePath, self.EPCType)
			sys.exit('Insufficient EPC Parameters')
		mySSH = SSH.SSHConnection() 
		mySSH.open(self.EPCIPAddress, self.EPCUserName, self.EPCPassword)
		if re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
			logging.debug('Using the OAI EPC Release 14 MME')
			mySSH.command('cd ' + self.EPCSourceCodePath + '/scripts', '\$', 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S rm -f mme_' + self.testCase_id + '.log', '\$', 5)
			mySSH.command('echo "./run_mme --config-file /usr/local/etc/oai/mme.conf --set-virt-if" > ./my-mme.sh', '\$', 5)
			mySSH.command('chmod 755 ./my-mme.sh', '\$', 5)
			mySSH.command('sudo daemon --unsafe --name=mme_daemon --chdir=' + self.EPCSourceCodePath + '/scripts -o ' + self.EPCSourceCodePath + '/scripts/mme_' + self.testCase_id + '.log ./my-mme.sh', '\$', 5)
		elif re.match('OAI', self.EPCType, re.IGNORECASE):
			mySSH.command('cd ' + self.EPCSourceCodePath, '\$', 5)
			mySSH.command('source oaienv', '\$', 5)
			mySSH.command('cd scripts', '\$', 5)
			mySSH.command('stdbuf -o0 hostname', '\$', 5)
			result = re.search('hostname\\\\r\\\\n(?P<host_name>[a-zA-Z0-9\-\_]+)\\\\r\\\\n', mySSH.getBefore())
			if result is None:
				logging.debug('\u001B[1;37;41m Hostname Not Found! \u001B[0m')
				sys.exit(1)
			host_name = result.group('host_name')
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S ./run_mme 2>&1 | stdbuf -o0 tee -a mme_' + self.testCase_id + '.log &', 'MME app initialization complete', 100)
		elif re.match('ltebox', self.EPCType, re.IGNORECASE):
			mySSH.command('cd /opt/ltebox/tools', '\$', 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S ./start_mme', '\$', 5)
		else:
			logging.error('This option should not occur!')
		mySSH.close()
		#self.CreateHtmlTestRow(self.EPCType, 'OK', ALL_PROCESSES_OK)

	def InitializeSPGW(self):
		if self.EPCIPAddress == '' or self.EPCUserName == '' or self.EPCPassword == '' or self.EPCSourceCodePath == '' or self.EPCType == '':
			HELP.GenericHelp(Version)
			HELP.EPCSrvHelp(self.EPCIPAddress, self.EPCUserName, self.EPCPassword, self.EPCSourceCodePath, self.EPCType)
			sys.exit('Insufficient EPC Parameters')
		mySSH = SSH.SSHConnection() 
		mySSH.open(self.EPCIPAddress, self.EPCUserName, self.EPCPassword)
		if re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
			logging.debug('Using the OAI EPC Release 14 SPGW-CUPS')
			mySSH.command('cd ' + self.EPCSourceCodePath + '/scripts', '\$', 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S rm -f spgwc_' + self.testCase_id + '.log spgwu_' + self.testCase_id + '.log', '\$', 5)
			mySSH.command('echo "spgwc -c /usr/local/etc/oai/spgw_c.conf" > ./my-spgwc.sh', '\$', 5)
			mySSH.command('chmod 755 ./my-spgwc.sh', '\$', 5)
			mySSH.command('sudo daemon --unsafe --name=spgwc_daemon --chdir=' + self.EPCSourceCodePath + '/scripts -o ' + self.EPCSourceCodePath + '/scripts/spgwc_' + self.testCase_id + '.log ./my-spgwc.sh', '\$', 5)
			time.sleep(5)
			mySSH.command('echo "spgwu -c /usr/local/etc/oai/spgw_u.conf" > ./my-spgwu.sh', '\$', 5)
			mySSH.command('chmod 755 ./my-spgwu.sh', '\$', 5)
			mySSH.command('sudo daemon --unsafe --name=spgwu_daemon --chdir=' + self.EPCSourceCodePath + '/scripts -o ' + self.EPCSourceCodePath + '/scripts/spgwu_' + self.testCase_id + '.log ./my-spgwu.sh', '\$', 5)
		elif re.match('OAI', self.EPCType, re.IGNORECASE):
			mySSH.command('cd ' + self.EPCSourceCodePath, '\$', 5)
			mySSH.command('source oaienv', '\$', 5)
			mySSH.command('cd scripts', '\$', 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S ./run_spgw 2>&1 | stdbuf -o0 tee -a spgw_' + self.testCase_id + '.log &', 'Initializing SPGW-APP task interface: DONE', 30)
		elif re.match('ltebox', self.EPCType, re.IGNORECASE):
			mySSH.command('cd /opt/ltebox/tools', '\$', 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S ./start_xGw', '\$', 5)
		else:
			logging.error('This option should not occur!')
		mySSH.close()
		#self.CreateHtmlTestRow(self.EPCType, 'OK', ALL_PROCESSES_OK)


	def CheckHSSProcess(self, status_queue):
		try:
			mySSH = SSH.SSHConnection() 
			mySSH.open(self.EPCIPAddress, self.EPCUserName, self.EPCPassword)
			mySSH.command('stdbuf -o0 ps -aux | grep --color=never hss | grep -v grep', '\$', 5)
			if re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
				result = re.search('oai_hss -j', mySSH.getBefore())
			elif re.match('OAI', self.EPCType, re.IGNORECASE):
				result = re.search('\/bin\/bash .\/run_', mySSH.getBefore())
			elif re.match('ltebox', self.EPCType, re.IGNORECASE):
				result = re.search('hss_sim s6as diam_hss', mySSH.getBefore())
			else:
				logging.error('This should not happen!')
			if result is None:
				logging.debug('\u001B[1;37;41m HSS Process Not Found! \u001B[0m')
				status_queue.put(CONST.HSS_PROCESS_FAILED)
			else:
				status_queue.put(CONST.HSS_PROCESS_OK)
			mySSH.close()
		except:
			os.kill(os.getppid(),signal.SIGUSR1)

	def CheckMMEProcess(self, status_queue):
		try:
			mySSH = SSH.SSHConnection() 
			mySSH.open(self.EPCIPAddress, self.EPCUserName, self.EPCPassword)
			mySSH.command('stdbuf -o0 ps -aux | grep --color=never mme | grep -v grep', '\$', 5)
			if re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
				result = re.search('mme -c', mySSH.getBefore())
			elif re.match('OAI', self.EPCType, re.IGNORECASE):
				result = re.search('\/bin\/bash .\/run_', mySSH.getBefore())
			elif re.match('ltebox', self.EPCType, re.IGNORECASE):
				result = re.search('mme', mySSH.getBefore())
			else:
				logging.error('This should not happen!')
			if result is None:
				logging.debug('\u001B[1;37;41m MME Process Not Found! \u001B[0m')
				status_queue.put(CONST.MME_PROCESS_FAILED)
			else:
				status_queue.put(CONST.MME_PROCESS_OK)
			mySSH.close()
		except:
			os.kill(os.getppid(),signal.SIGUSR1)

	def CheckSPGWProcess(self, status_queue):
		try:
			mySSH = SSH.SSHConnection() 
			mySSH.open(self.EPCIPAddress, self.EPCUserName, self.EPCPassword)
			if re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
				mySSH.command('stdbuf -o0 ps -aux | grep --color=never spgw | grep -v grep', '\$', 5)
				result = re.search('spgwu -c ', mySSH.getBefore())
			elif re.match('OAI', self.EPCType, re.IGNORECASE):
				mySSH.command('stdbuf -o0 ps -aux | grep --color=never spgw | grep -v grep', '\$', 5)
				result = re.search('\/bin\/bash .\/run_', mySSH.getBefore())
			elif re.match('ltebox', self.EPCType, re.IGNORECASE):
				mySSH.command('stdbuf -o0 ps -aux | grep --color=never xGw | grep -v grep', '\$', 5)
				result = re.search('xGw', mySSH.getBefore())
			else:
				logging.error('This should not happen!')
			if result is None:
				logging.debug('\u001B[1;37;41m SPGW Process Not Found! \u001B[0m')
				status_queue.put(CONST.SPGW_PROCESS_FAILED)
			else:
				status_queue.put(CONST.SPGW_PROCESS_OK)
			mySSH.close()
		except:
			os.kill(os.getppid(),signal.SIGUSR1)


	def TerminateHSS(self):
		mySSH = SSH.SSHConnection() 
		mySSH.open(self.EPCIPAddress, self.EPCUserName, self.EPCPassword)
		if re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S killall --signal SIGINT oai_hss || true', '\$', 5)
			time.sleep(2)
			mySSH.command('stdbuf -o0  ps -aux | grep hss | grep -v grep', '\$', 5)
			result = re.search('oai_hss -j', mySSH.getBefore())
			if result is not None:
				mySSH.command('echo ' + self.EPCPassword + ' | sudo -S killall --signal SIGKILL oai_hss || true', '\$', 5)
			mySSH.command('rm -f ' + self.EPCSourceCodePath + '/scripts/my-hss.sh', '\$', 5)
		elif re.match('OAI', self.EPCType, re.IGNORECASE):
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S killall --signal SIGINT run_hss oai_hss || true', '\$', 5)
			time.sleep(2)
			mySSH.command('stdbuf -o0  ps -aux | grep hss | grep -v grep', '\$', 5)
			result = re.search('\/bin\/bash .\/run_', mySSH.getBefore())
			if result is not None:
				mySSH.command('echo ' + self.EPCPassword + ' | sudo -S killall --signal SIGKILL run_hss oai_hss || true', '\$', 5)
		elif re.match('ltebox', self.EPCType, re.IGNORECASE):
			mySSH.command('cd ' + self.EPCSourceCodePath, '\$', 5)
			mySSH.command('cd scripts', '\$', 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S daemon --name=simulated_hss --stop', '\$', 5)
			time.sleep(1)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S killall --signal SIGKILL hss_sim', '\$', 5)
		else:
			logging.error('This should not happen!')
		mySSH.close()
		#self.CreateHtmlTestRow('N/A', 'OK', CONST.ALL_PROCESSES_OK)

	def TerminateMME(self):
		mySSH = SSH.SSHConnection() 
		mySSH.open(self.EPCIPAddress, self.EPCUserName, self.EPCPassword)
		if re.match('OAI', self.EPCType, re.IGNORECASE) or re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S killall --signal SIGINT run_mme mme || true', '\$', 5)
			time.sleep(2)
			mySSH.command('stdbuf -o0 ps -aux | grep mme | grep -v grep', '\$', 5)
			result = re.search('mme -c', mySSH.getBefore())
			if result is not None:
				mySSH.command('echo ' + self.EPCPassword + ' | sudo -S killall --signal SIGKILL run_mme mme || true', '\$', 5)
			mySSH.command('rm -f ' + self.EPCSourceCodePath + '/scripts/my-mme.sh', '\$', 5)
		elif re.match('ltebox', self.EPCType, re.IGNORECASE):
			mySSH.command('cd /opt/ltebox/tools', '\$', 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S ./stop_mme', '\$', 5)
		else:
			logging.error('This should not happen!')
		mySSH.close()
		#self.CreateHtmlTestRow('N/A', 'OK', ALL_PROCESSES_OK)

	def TerminateSPGW(self):
		mySSH = SSH.SSHConnection() 
		mySSH.open(self.EPCIPAddress, self.EPCUserName, self.EPCPassword)
		if re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S killall --signal SIGINT spgwc spgwu || true', '\$', 5)
			time.sleep(2)
			mySSH.command('stdbuf -o0 ps -aux | grep spgw | grep -v grep', '\$', 5)
			result = re.search('spgwc -c |spgwu -c ', mySSH.getBefore())
			if result is not None:
				mySSH.command('echo ' + self.EPCPassword + ' | sudo -S killall --signal SIGKILL spgwc spgwu || true', '\$', 5)
			mySSH.command('rm -f ' + self.EPCSourceCodePath + '/scripts/my-spgw*.sh', '\$', 5)
			mySSH.command('stdbuf -o0 ps -aux | grep tshark | grep -v grep', '\$', 5)
			result = re.search('-w ', mySSH.getBefore())
			if result is not None:
				mySSH.command('echo ' + self.EPCPassword + ' | sudo -S killall --signal SIGINT tshark || true', '\$', 5)
				mySSH.command('echo ' + self.EPCPassword + ' | sudo -S chmod 666 ' + self.EPCSourceCodePath + '/scripts/*.pcap', '\$', 5)
		elif re.match('OAI', self.EPCType, re.IGNORECASE):
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S killall --signal SIGINT run_spgw spgw || true', '\$', 5)
			time.sleep(2)
			mySSH.command('stdbuf -o0 ps -aux | grep spgw | grep -v grep', '\$', 5)
			result = re.search('\/bin\/bash .\/run_', mySSH.getBefore())
			if result is not None:
				mySSH.command('echo ' + self.EPCPassword + ' | sudo -S killall --signal SIGKILL run_spgw spgw || true', '\$', 5)
		elif re.match('ltebox', self.EPCType, re.IGNORECASE):
			mySSH.command('cd /opt/ltebox/tools', '\$', 5)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S ./stop_xGw', '\$', 5)
		else:
			logging.error('This should not happen!')
		mySSH.close()
		#self.CreateHtmlTestRow('N/A', 'OK', CONST.ALL_PROCESSES_OK)


	def LogCollectHSS(self):
		mySSH = SSH.SSHConnection() 
		mySSH.open(self.EPCIPAddress, self.EPCUserName, self.EPCPassword)
		mySSH.command('cd ' + self.EPCSourceCodePath + '/scripts', '\$', 5)
		mySSH.command('rm -f hss.log.zip', '\$', 5)
		if re.match('OAI', self.EPCType, re.IGNORECASE) or re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
			mySSH.command('zip hss.log.zip hss*.log', '\$', 60)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S rm hss*.log', '\$', 5)
			if re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
				mySSH.command('zip hss.log.zip logs/hss*.* *.pcap', '\$', 60)
				mySSH.command('echo ' + self.EPCPassword + ' | sudo -S rm -f logs/hss*.* *.pcap', '\$', 5)
		elif re.match('ltebox', self.EPCType, re.IGNORECASE):
			mySSH.command('cp /opt/hss_sim0609/hss.log .', '\$', 60)
			mySSH.command('zip hss.log.zip hss.log', '\$', 60)
		else:
			logging.error('This option should not occur!')
		mySSH.close()

	def LogCollectMME(self):
		mySSH = SSH.SSHConnection() 
		mySSH.open(self.EPCIPAddress, self.EPCUserName, self.EPCPassword)
		mySSH.command('cd ' + self.EPCSourceCodePath + '/scripts', '\$', 5)
		mySSH.command('rm -f mme.log.zip', '\$', 5)
		if re.match('OAI', self.EPCType, re.IGNORECASE) or re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
			mySSH.command('zip mme.log.zip mme*.log', '\$', 60)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S rm mme*.log', '\$', 5)
		elif re.match('ltebox', self.EPCType, re.IGNORECASE):
			mySSH.command('cp /opt/ltebox/var/log/*Log.0 .', '\$', 5)
			mySSH.command('zip mme.log.zip mmeLog.0 s1apcLog.0 s1apsLog.0 s11cLog.0 libLog.0 s1apCodecLog.0', '\$', 60)
		else:
			logging.error('This option should not occur!')
		mySSH.close()

	def LogCollectSPGW(self):
		mySSH = SSH.SSHConnection() 
		mySSH.open(self.EPCIPAddress, self.EPCUserName, self.EPCPassword)
		mySSH.command('cd ' + self.EPCSourceCodePath + '/scripts', '\$', 5)
		mySSH.command('rm -f spgw.log.zip', '\$', 5)
		if re.match('OAI', self.EPCType, re.IGNORECASE) or re.match('OAI-Rel14-CUPS', self.EPCType, re.IGNORECASE):
			mySSH.command('zip spgw.log.zip spgw*.log', '\$', 60)
			mySSH.command('echo ' + self.EPCPassword + ' | sudo -S rm spgw*.log', '\$', 5)
		elif re.match('ltebox', self.EPCType, re.IGNORECASE):
			mySSH.command('cp /opt/ltebox/var/log/xGwLog.0 .', '\$', 5)
			mySSH.command('zip spgw.log.zip xGwLog.0', '\$', 60)
		else:
			logging.error('This option should not occur!')
		mySSH.close()

