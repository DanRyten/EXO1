<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="20008000">
	<Item Name="New Folder" Type="Target Folder">
		<Item Name="roboRIO-330-FRC" Type="RT roboRIO">
			<Property Name="alias.name" Type="Str">roboRIO-330-FRC</Property>
			<Property Name="alias.value" Type="Str">172.22.11.2</Property>
			<Property Name="CCSymbols" Type="Str">TARGET_TYPE,RT;OS,Linux;CPU,ARM;DeviceCode,76F2;</Property>
			<Property Name="crio.ControllerPID" Type="Str">76F2</Property>
			<Property Name="host.ResponsivenessCheckEnabled" Type="Bool">true</Property>
			<Property Name="host.ResponsivenessCheckPingDelay" Type="UInt">5000</Property>
			<Property Name="host.ResponsivenessCheckPingTimeout" Type="UInt">1000</Property>
			<Property Name="host.TargetCPUID" Type="UInt">8</Property>
			<Property Name="host.TargetOSID" Type="UInt">8</Property>
			<Property Name="NI.SortType" Type="Int">3</Property>
			<Property Name="target.cleanupVisa" Type="Bool">false</Property>
			<Property Name="target.FPProtocolGlobals_ControlTimeLimit" Type="Int">300</Property>
			<Property Name="target.getDefault-&gt;WebServer.Port" Type="Int">80</Property>
			<Property Name="target.getDefault-&gt;WebServer.Timeout" Type="Int">60</Property>
			<Property Name="target.IOScan.Faults" Type="Str"></Property>
			<Property Name="target.IOScan.NetVarPeriod" Type="UInt">100</Property>
			<Property Name="target.IOScan.NetWatchdogEnabled" Type="Bool">false</Property>
			<Property Name="target.IOScan.Period" Type="UInt">10000</Property>
			<Property Name="target.IOScan.PowerupMode" Type="UInt">0</Property>
			<Property Name="target.IOScan.Priority" Type="UInt">0</Property>
			<Property Name="target.IOScan.ReportModeConflict" Type="Bool">true</Property>
			<Property Name="target.IsRemotePanelSupported" Type="Bool">true</Property>
			<Property Name="target.RTCPULoadMonitoringEnabled" Type="Bool">true</Property>
			<Property Name="target.RTDebugWebServerHTTPPort" Type="Int">8001</Property>
			<Property Name="target.RTTarget.ApplicationPath" Type="Path">/c/ni-rt/startup/startup.rtexe</Property>
			<Property Name="target.RTTarget.EnableFileSharing" Type="Bool">true</Property>
			<Property Name="target.RTTarget.IPAccess" Type="Str">+*</Property>
			<Property Name="target.RTTarget.LaunchAppAtBoot" Type="Bool">false</Property>
			<Property Name="target.RTTarget.VIPath" Type="Path">/home/lvuser/natinst/bin</Property>
			<Property Name="target.server.app.propertiesEnabled" Type="Bool">true</Property>
			<Property Name="target.server.control.propertiesEnabled" Type="Bool">true</Property>
			<Property Name="target.server.tcp.access" Type="Str">+*</Property>
			<Property Name="target.server.tcp.enabled" Type="Bool">false</Property>
			<Property Name="target.server.tcp.paranoid" Type="Bool">true</Property>
			<Property Name="target.server.tcp.port" Type="Int">3363</Property>
			<Property Name="target.server.tcp.serviceName" Type="Str">Main Application Instance/VI Server</Property>
			<Property Name="target.server.tcp.serviceName.default" Type="Str">Main Application Instance/VI Server</Property>
			<Property Name="target.server.vi.access" Type="Str">+*</Property>
			<Property Name="target.server.vi.callsEnabled" Type="Bool">true</Property>
			<Property Name="target.server.vi.propertiesEnabled" Type="Bool">true</Property>
			<Property Name="target.WebServer.Config" Type="Str">Listen 8000

NI.ServerName default
DocumentRoot "$LVSERVER_DOCROOT"
TypesConfig "$LVSERVER_CONFIGROOT/mime.types"
DirectoryIndex index.htm
WorkerLimit 10
InactivityTimeout 60

LoadModulePath "$LVSERVER_MODULEPATHS"
LoadModule LVAuth lvauthmodule
LoadModule LVRFP lvrfpmodule

#
# Pipeline Definition
#

SetConnector netConnector

AddHandler LVAuth
AddHandler LVRFP

AddHandler fileHandler ""

AddOutputFilter chunkFilter


</Property>
			<Property Name="target.WebServer.Enabled" Type="Bool">false</Property>
			<Property Name="target.WebServer.LogEnabled" Type="Bool">false</Property>
			<Property Name="target.WebServer.LogPath" Type="Path">/c/ni-rt/system/www/www.log</Property>
			<Property Name="target.WebServer.Port" Type="Int">80</Property>
			<Property Name="target.WebServer.RootPath" Type="Path">/c/ni-rt/system/www</Property>
			<Property Name="target.WebServer.TcpAccess" Type="Str">c+*</Property>
			<Property Name="target.WebServer.Timeout" Type="Int">60</Property>
			<Property Name="target.WebServer.ViAccess" Type="Str">+*</Property>
			<Property Name="target.webservices.SecurityAPIKey" Type="Str">PqVr/ifkAQh+lVrdPIykXlFvg12GhhQFR8H9cUhphgg=:pTe9HRlQuMfJxAG6QCGq7UvoUpJzAzWGKy5SbZ+roSU=</Property>
			<Property Name="target.webservices.ValidTimestampWindow" Type="Int">15</Property>
			<Item Name="NI-Embedded CAN" Type="Folder">
				<Item Name="subVIs" Type="Folder">
					<Item Name="NI-Embedded CAN for RIO subVI - CAN Frame to Table.vi" Type="VI" URL="../../../../../../../Program Files (x86)/National Instruments/LabVIEW 2020/examples/NI-Embedded CAN for RIO/NI-Embedded CAN for RIO subVI - CAN Frame to Table.vi"/>
				</Item>
				<Item Name="CAN Frame Input Output - single port.vi" Type="VI" URL="../../../../../../../Program Files (x86)/National Instruments/LabVIEW 2020/examples/NI-Embedded CAN for RIO/CAN Frame Input Output - single port.vi"/>
				<Item Name="CAN Frame Input Stream - with hardware filter.vi" Type="VI" URL="../../../../../../../Program Files (x86)/National Instruments/LabVIEW 2020/examples/NI-Embedded CAN for RIO/CAN Frame Input Stream - with hardware filter.vi"/>
				<Item Name="CAN Frame Input Stream.vi" Type="VI" URL="../../../../../../../Program Files (x86)/National Instruments/LabVIEW 2020/examples/NI-Embedded CAN for RIO/CAN Frame Input Stream.vi"/>
				<Item Name="CAN Frame Output - event based.vi" Type="VI" URL="../../../../../../../Program Files (x86)/National Instruments/LabVIEW 2020/examples/NI-Embedded CAN for RIO/CAN Frame Output - event based.vi"/>
				<Item Name="CAN Frame Output Stream.vi" Type="VI" URL="../../../../../../../Program Files (x86)/National Instruments/LabVIEW 2020/examples/NI-Embedded CAN for RIO/CAN Frame Output Stream.vi"/>
				<Item Name="Transceiver Sleep Mode.vi" Type="VI" URL="../../../../../../../Program Files (x86)/National Instruments/LabVIEW 2020/examples/NI-Embedded CAN for RIO/Transceiver Sleep Mode.vi"/>
			</Item>
			<Item Name="support - acquire data.vi" Type="VI" URL="/&lt;resource&gt;/plugins/NewDialogFiles/ProjectWizards/RTNPA/New Projects/support - acquire data.vi"/>
			<Item Name="target - single rate.vi" Type="VI" URL="../target - single rate.vi"/>
			<Item Name="Chassis" Type="roboRIO Chassis">
				<Property Name="crio.ProgrammingMode" Type="Str">fpga</Property>
				<Property Name="crio.ResourceID" Type="Str">RIO0</Property>
				<Property Name="crio.Type" Type="Str">roboRIO</Property>
				<Property Name="NI.SortType" Type="Int">3</Property>
				<Item Name="Real-Time Scan Resources" Type="Module Container">
					<Property Name="crio.ModuleContainerType" Type="Str">crio.RSIModuleContainer</Property>
				</Item>
				<Item Name="FPGA Target" Type="FPGA Target">
					<Property Name="AutoRun" Type="Bool">false</Property>
					<Property Name="configString.guid" Type="Str">{03B4C7F9-31B1-4375-8D6C-3C498D3E98C8}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{046BD9D1-BCD3-41D3-8623-714B59353BDD}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{052DD952-8FDA-4F77-85A3-7A41612B9001}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{0AC976C2-1F45-4046-B2F8-2E288E6D2005}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{0D16B38B-CA32-4EE9-9710-6FD655BEAF8F}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{12514A7D-61D9-4C37-BC1B-162DEB9E67B2}resource=/MXP/AO0;0;WriteMethodType=U16{158DD48D-1737-41BE-B4EC-E66B4BDFE83C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{175C3D9A-F468-4331-8CAB-728A1FDA27B2}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{230B38D1-8B56-4451-855E-55F01B0EC059}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{244FAFD2-C3E0-49D6-A3E4-BD737A67ACDE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{252CD302-84F3-4E3E-92B6-5E5046DF55F1}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{2728990C-0955-4ABE-A595-1BA7CAC22012}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{27C15163-E4EE-44B0-8B99-BF3C328C3ACB}resource=/AccelerationX;0;ReadMethodType=I16{2897DBE3-A873-4026-AEA6-A38F1A626F13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{29DFD2E7-77EB-4E36-AA74-30AE988228F0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{2C9625CB-20E2-4237-9585-44C7D17B5578}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{2F22D623-CACF-4A67-9174-08942B417B0D}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{32D8B85C-4EC9-4FDB-83BB-B94383281261}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{3DD6FEA2-C30B-443B-A2DE-D511BB5E9915}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3FC14A16-BBBF-4C0B-9B72-A62A15D2A586}resource=/AccelerationZ;0;ReadMethodType=I16{48613830-9912-4349-BD50-2B27E694943B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{4CEAD8BB-FECF-4DEC-8524-E9FAA1C4B84D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{4EF86A8A-1ED2-4212-A38A-871AA0AC9DED}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{5021F2D0-2FEB-415A-A236-C91403C8C434}resource=/MXP/AO1;0;WriteMethodType=U16{553801DF-5D29-48CA-B5A5-F856F4279932}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{56318075-D11F-4659-AC03-D0751FEB1226}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{5C1B663D-FAF4-4831-BBE6-051C45990D7F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{5C3973A4-8F31-4AE4-9FAA-E0CA91B10F51}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{61A1B603-4F86-4DC9-9B6D-426B7EF107C8}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{6255D830-D90A-4488-A943-B5B3382F7B1A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{634DC4D4-541F-420E-AB3D-9984B466CBDF}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{663F166E-1631-4874-BA76-C54FBBE99E51}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{6DBFE461-A262-4359-9695-7067BBC00DBF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{71715594-4816-4D85-AB80-179F18D313CB}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{717E653E-4F6B-4A16-93C9-215929314605}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{79152894-ACE9-4CAF-B127-5C7C0A84F8A4}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{7E1B5D5C-CA6B-44B5-B7F3-E573115FE528}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{7F125321-43CF-4483-A3FC-F4DEB1914881}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{867E4EEB-23A7-4071-B170-8A90A02FEDFB}resource=/AnalogIn1;0;ReadMethodType=U16{86D88082-2D6C-47CD-9194-EDCF3A80AF61}resource=/AnalogIn3;0;ReadMethodType=U16{8810D240-93E6-4B16-BAF3-CFEEFAB8DC43}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{88A8C9B0-92CD-4926-B733-E8991C0D5DC6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{8AA6E36B-0E35-4B83-AD09-2E5987794A75}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{94B523E0-ABDB-482E-B3B6-BDC9DA454042}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{982D32E9-2FCB-4A4B-8D8B-18BE3924B5CF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{98DBF425-91B8-4CF9-8E96-EA9A7E6662F5}resource=/MXP/AI3;0;ReadMethodType=U16{995E239E-AECE-4BAD-B0DC-D9C51A83AF55}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{9F00E1FF-FAD7-4B26-81F7-0FD4CFAE6535}resource=/AnalogIn2;0;ReadMethodType=U16{A0D01AB8-BB91-4375-B91F-EFBC1E1D0A5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{A65F0CFC-BEF4-4F03-8A0A-3D6755976458}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{ADD767DC-89F6-4E99-A999-C3C511E8F049}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B186E5B0-13AF-4F67-8DCD-5CFB0E38B8CD}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{B40402A8-6BB2-410E-BE83-269307075AA4}resource=/MXP/AI1;0;ReadMethodType=U16{B59FBA04-C081-4910-979C-F475BF94B4E7}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{B6D79384-1544-48F5-8C30-015ACFAD2B7D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{BA2B9DA1-3804-44F4-B42F-B944367F4830}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB927CDD-6CEA-4FA3-BA27-F8B37B322B5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{C136E580-FDC3-4FD0-A73E-8FADF50FCEF1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{C832ECE7-8171-418B-A4A0-368DF1088B67}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{C86B5E87-7DB7-48BF-AD9F-3A17A56A19EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{CBB1C1E1-B5D6-43E3-81A4-2CE233E8F521}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{D0188368-5943-4B71-A85B-0B5C16AF1080}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{D1EBD955-AB5D-45B8-8908-322BF3D03143}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{D76A6B63-78CF-4DE7-ABC9-110AAC617555}resource=/MXP/AI0;0;ReadMethodType=U16{D9B2ADEF-4B0F-4EB1-9D0F-509EE0B9567C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{DAE46D39-D9D1-4089-8FC2-5135206CE1EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{DC25416B-85BC-4736-B363-87C87591B48D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{DF69F36E-FFFB-4674-859E-4CC1FA6F5787}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{E8D2FB32-9FA4-4993-984C-CB8CB4AC97E7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{EAF2C81F-5F74-4BC2-AE3D-33B3D5D9DD0B}resource=/AnalogIn0;0;ReadMethodType=U16{EB635694-8D5D-4B46-8BD0-5DC72292FF44}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{ED3CA370-9D51-46A4-A6BF-1D32C231286B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1AC4ABC-9F5A-430A-B8B7-ADC575BE3272}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1E9396B-8491-4765-A5F0-90494714CF93}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{F9B030D6-994B-4D02-8DD1-B94488513E67}resource=/AccelerationY;0;ReadMethodType=I16{FB0D33ED-4CE1-49A3-853E-099CFA00F792}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{FC2A8B33-1B74-4A37-A867-88C76E8C298B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{FF9337F1-C277-47DF-9AAC-D1D389DC8EA4}resource=/MXP/AI2;0;ReadMethodType=U16roboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
					<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
					<Property Name="NI.LV.FPGA.CompileConfigString" Type="Str">roboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
					<Property Name="NI.LV.FPGA.Version" Type="Int">6</Property>
					<Property Name="Resource Name" Type="Str">RIO0</Property>
					<Property Name="Target Class" Type="Str">roboRIO</Property>
					<Property Name="Top-Level Timing Source" Type="Str">40 MHz Onboard Clock</Property>
					<Property Name="Top-Level Timing Source Is Default" Type="Bool">true</Property>
					<Item Name="Onboard I/O" Type="Folder">
						<Item Name="LED" Type="Folder">
							<Item Name="Radio LED Green" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Radio LED Green</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{230B38D1-8B56-4451-855E-55F01B0EC059}</Property>
							</Item>
							<Item Name="Radio LED Red" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Radio LED Red</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{BB927CDD-6CEA-4FA3-BA27-F8B37B322B5A}</Property>
							</Item>
							<Item Name="Comm LED Green" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Comm LED Green</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{8810D240-93E6-4B16-BAF3-CFEEFAB8DC43}</Property>
							</Item>
							<Item Name="Comm LED Red" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Comm LED Red</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{C86B5E87-7DB7-48BF-AD9F-3A17A56A19EB}</Property>
							</Item>
							<Item Name="Mode LED Green" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Mode LED Green</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{5C3973A4-8F31-4AE4-9FAA-E0CA91B10F51}</Property>
							</Item>
							<Item Name="Mode LED Red" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Mode LED Red</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{995E239E-AECE-4BAD-B0DC-D9C51A83AF55}</Property>
							</Item>
						</Item>
						<Item Name="System Reset" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/System Reset</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{79152894-ACE9-4CAF-B127-5C7C0A84F8A4}</Property>
						</Item>
						<Item Name="RSL" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/RSL</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{E8D2FB32-9FA4-4993-984C-CB8CB4AC97E7}</Property>
						</Item>
						<Item Name="User Button" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/User Button</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{553801DF-5D29-48CA-B5A5-F856F4279932}</Property>
						</Item>
						<Item Name="AccelerationX" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AccelerationX</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{27C15163-E4EE-44B0-8B99-BF3C328C3ACB}</Property>
						</Item>
						<Item Name="AccelerationY" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AccelerationY</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{F9B030D6-994B-4D02-8DD1-B94488513E67}</Property>
						</Item>
						<Item Name="AccelerationZ" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AccelerationZ</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{3FC14A16-BBBF-4C0B-9B72-A62A15D2A586}</Property>
						</Item>
					</Item>
					<Item Name="Analog" Type="Folder">
						<Item Name="AnalogIn0" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AnalogIn0</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{EAF2C81F-5F74-4BC2-AE3D-33B3D5D9DD0B}</Property>
						</Item>
						<Item Name="AnalogIn1" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AnalogIn1</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{867E4EEB-23A7-4071-B170-8A90A02FEDFB}</Property>
						</Item>
						<Item Name="AnalogIn2" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AnalogIn2</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{9F00E1FF-FAD7-4B26-81F7-0FD4CFAE6535}</Property>
						</Item>
						<Item Name="AnalogIn3" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AnalogIn3</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{86D88082-2D6C-47CD-9194-EDCF3A80AF61}</Property>
						</Item>
					</Item>
					<Item Name="I2C" Type="Folder">
						<Item Name="I2C SDA" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/I2C SDA</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{C136E580-FDC3-4FD0-A73E-8FADF50FCEF1}</Property>
						</Item>
						<Item Name="I2C SCL" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/I2C SCL</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{7E1B5D5C-CA6B-44B5-B7F3-E573115FE528}</Property>
						</Item>
					</Item>
					<Item Name="SPI" Type="Folder">
						<Item Name="SPI CLK" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/SPI CLK</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{B59FBA04-C081-4910-979C-F475BF94B4E7}</Property>
						</Item>
						<Item Name="SPI MOSI" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/SPI MOSI</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{71715594-4816-4D85-AB80-179F18D313CB}</Property>
						</Item>
						<Item Name="SPI MISO" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/SPI MISO</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{DC25416B-85BC-4736-B363-87C87591B48D}</Property>
						</Item>
						<Item Name="SPI CS0" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/SPI CS0</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{0D16B38B-CA32-4EE9-9710-6FD655BEAF8F}</Property>
						</Item>
						<Item Name="SPI CS1" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/SPI CS1</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{56318075-D11F-4659-AC03-D0751FEB1226}</Property>
						</Item>
						<Item Name="SPI CS2" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/SPI CS2</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{2728990C-0955-4ABE-A595-1BA7CAC22012}</Property>
						</Item>
						<Item Name="SPI CS3" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/SPI CS3</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{7F125321-43CF-4483-A3FC-F4DEB1914881}</Property>
						</Item>
					</Item>
					<Item Name="PWM" Type="Folder">
						<Item Name="PWM/PWM0" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/PWM/PWM0</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{03B4C7F9-31B1-4375-8D6C-3C498D3E98C8}</Property>
						</Item>
						<Item Name="PWM/PWM1" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/PWM/PWM1</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{0AC976C2-1F45-4046-B2F8-2E288E6D2005}</Property>
						</Item>
						<Item Name="PWM/PWM2" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/PWM/PWM2</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{6255D830-D90A-4488-A943-B5B3382F7B1A}</Property>
						</Item>
						<Item Name="PWM/PWM3" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/PWM/PWM3</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{EB635694-8D5D-4B46-8BD0-5DC72292FF44}</Property>
						</Item>
						<Item Name="PWM/PWM4" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/PWM/PWM4</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{175C3D9A-F468-4331-8CAB-728A1FDA27B2}</Property>
						</Item>
						<Item Name="PWM/PWM5" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/PWM/PWM5</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{252CD302-84F3-4E3E-92B6-5E5046DF55F1}</Property>
						</Item>
						<Item Name="PWM/PWM6" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/PWM/PWM6</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{717E653E-4F6B-4A16-93C9-215929314605}</Property>
						</Item>
						<Item Name="PWM/PWM7" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/PWM/PWM7</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{2F22D623-CACF-4A67-9174-08942B417B0D}</Property>
						</Item>
						<Item Name="PWM/PWM8" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/PWM/PWM8</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{94B523E0-ABDB-482E-B3B6-BDC9DA454042}</Property>
						</Item>
						<Item Name="PWM/PWM9" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/PWM/PWM9</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{DAE46D39-D9D1-4089-8FC2-5135206CE1EB}</Property>
						</Item>
					</Item>
					<Item Name="Relay" Type="Folder">
						<Item Name="RelayFwd0" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/RelayFwd0</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{FB0D33ED-4CE1-49A3-853E-099CFA00F792}</Property>
						</Item>
						<Item Name="RelayFwd1" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/RelayFwd1</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{32D8B85C-4EC9-4FDB-83BB-B94383281261}</Property>
						</Item>
						<Item Name="RelayFwd2" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/RelayFwd2</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{982D32E9-2FCB-4A4B-8D8B-18BE3924B5CF}</Property>
						</Item>
						<Item Name="RelayFwd3" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/RelayFwd3</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{D0188368-5943-4B71-A85B-0B5C16AF1080}</Property>
						</Item>
						<Item Name="RelayRev0" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/RelayRev0</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{A0D01AB8-BB91-4375-B91F-EFBC1E1D0A5A}</Property>
						</Item>
						<Item Name="RelayRev1" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/RelayRev1</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{046BD9D1-BCD3-41D3-8623-714B59353BDD}</Property>
						</Item>
						<Item Name="RelayRev2" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/RelayRev2</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{6DBFE461-A262-4359-9695-7067BBC00DBF}</Property>
						</Item>
						<Item Name="RelayRev3" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/RelayRev3</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{5C1B663D-FAF4-4831-BBE6-051C45990D7F}</Property>
						</Item>
					</Item>
					<Item Name="DIO" Type="Folder">
						<Item Name="DIO/DIO0" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/DIO/DIO0</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{4CEAD8BB-FECF-4DEC-8524-E9FAA1C4B84D}</Property>
						</Item>
						<Item Name="DIO/DIO1" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/DIO/DIO1</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{F1AC4ABC-9F5A-430A-B8B7-ADC575BE3272}</Property>
						</Item>
						<Item Name="DIO/DIO2" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/DIO/DIO2</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{CBB1C1E1-B5D6-43E3-81A4-2CE233E8F521}</Property>
						</Item>
						<Item Name="DIO/DIO3" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/DIO/DIO3</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{634DC4D4-541F-420E-AB3D-9984B466CBDF}</Property>
						</Item>
						<Item Name="DIO/DIO4" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/DIO/DIO4</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{663F166E-1631-4874-BA76-C54FBBE99E51}</Property>
						</Item>
						<Item Name="DIO/DIO5" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/DIO/DIO5</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{BA2B9DA1-3804-44F4-B42F-B944367F4830}</Property>
						</Item>
						<Item Name="DIO/DIO6" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/DIO/DIO6</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{88A8C9B0-92CD-4926-B733-E8991C0D5DC6}</Property>
						</Item>
						<Item Name="DIO/DIO7" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/DIO/DIO7</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{D1EBD955-AB5D-45B8-8908-322BF3D03143}</Property>
						</Item>
						<Item Name="DIO/DIO8" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/DIO/DIO8</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{48613830-9912-4349-BD50-2B27E694943B}</Property>
						</Item>
						<Item Name="DIO/DIO9" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/DIO/DIO9</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{052DD952-8FDA-4F77-85A3-7A41612B9001}</Property>
						</Item>
					</Item>
					<Item Name="MXP" Type="Folder">
						<Item Name="Analog" Type="Folder">
							<Item Name="MXP/AO0" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/MXP/AO0</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{12514A7D-61D9-4C37-BC1B-162DEB9E67B2}</Property>
							</Item>
							<Item Name="MXP/AO1" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/MXP/AO1</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{5021F2D0-2FEB-415A-A236-C91403C8C434}</Property>
							</Item>
							<Item Name="MXP/AI0" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/MXP/AI0</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{D76A6B63-78CF-4DE7-ABC9-110AAC617555}</Property>
							</Item>
							<Item Name="MXP/AI1" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/MXP/AI1</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{B40402A8-6BB2-410E-BE83-269307075AA4}</Property>
							</Item>
							<Item Name="MXP/AI2" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/MXP/AI2</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{FF9337F1-C277-47DF-9AAC-D1D389DC8EA4}</Property>
							</Item>
							<Item Name="MXP/AI3" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/MXP/AI3</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{98DBF425-91B8-4CF9-8E96-EA9A7E6662F5}</Property>
							</Item>
						</Item>
						<Item Name="DIO7:0" Type="Folder">
							<Item Name="MXP/DIO0" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO0</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{FC2A8B33-1B74-4A37-A867-88C76E8C298B}</Property>
							</Item>
							<Item Name="MXP/DIO1" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO1</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{ED3CA370-9D51-46A4-A6BF-1D32C231286B}</Property>
							</Item>
							<Item Name="MXP/DIO2" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO2</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{61A1B603-4F86-4DC9-9B6D-426B7EF107C8}</Property>
							</Item>
							<Item Name="MXP/DIO3" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO3</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{D9B2ADEF-4B0F-4EB1-9D0F-509EE0B9567C}</Property>
							</Item>
							<Item Name="MXP/DIO4" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO4</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{F1E9396B-8491-4765-A5F0-90494714CF93}</Property>
							</Item>
							<Item Name="MXP/DIO5" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO5</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{8AA6E36B-0E35-4B83-AD09-2E5987794A75}</Property>
							</Item>
							<Item Name="MXP/DIO6" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO6</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{C832ECE7-8171-418B-A4A0-368DF1088B67}</Property>
							</Item>
							<Item Name="MXP/DIO7" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO7</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{29DFD2E7-77EB-4E36-AA74-30AE988228F0}</Property>
							</Item>
							<Item Name="MXP/DIO7:0" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO7:0</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{244FAFD2-C3E0-49D6-A3E4-BD737A67ACDE}</Property>
							</Item>
						</Item>
						<Item Name="DIO15:8" Type="Folder">
							<Item Name="MXP/DIO8" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO8</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{B186E5B0-13AF-4F67-8DCD-5CFB0E38B8CD}</Property>
							</Item>
							<Item Name="MXP/DIO9" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO9</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{ADD767DC-89F6-4E99-A999-C3C511E8F049}</Property>
							</Item>
							<Item Name="MXP/DIO10" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO10</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{3DD6FEA2-C30B-443B-A2DE-D511BB5E9915}</Property>
							</Item>
							<Item Name="MXP/DIO11" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO11</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{4EF86A8A-1ED2-4212-A38A-871AA0AC9DED}</Property>
							</Item>
							<Item Name="MXP/DIO12" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO12</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{B6D79384-1544-48F5-8C30-015ACFAD2B7D}</Property>
							</Item>
							<Item Name="MXP/DIO13" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO13</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{DF69F36E-FFFB-4674-859E-4CC1FA6F5787}</Property>
							</Item>
							<Item Name="MXP/DIO14" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO14</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{158DD48D-1737-41BE-B4EC-E66B4BDFE83C}</Property>
							</Item>
							<Item Name="MXP/DIO15" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO15</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{2897DBE3-A873-4026-AEA6-A38F1A626F13}</Property>
							</Item>
							<Item Name="MXP/DIO15:8" Type="Elemental IO">
								<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="ArbitrationForOutputEnable">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputEnable">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/MXP/DIO15:8</Value>
   </Attribute>
</AttributeSet>
</Property>
								<Property Name="FPGA.PersistentID" Type="Str">{2C9625CB-20E2-4237-9585-44C7D17B5578}</Property>
							</Item>
						</Item>
					</Item>
					<Item Name="EXO1_VI" Type="Folder">
						<Item Name="EXO1.vi" Type="VI" URL="../EXO1.vi">
							<Property Name="configString.guid" Type="Str">{03B4C7F9-31B1-4375-8D6C-3C498D3E98C8}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{046BD9D1-BCD3-41D3-8623-714B59353BDD}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{052DD952-8FDA-4F77-85A3-7A41612B9001}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{0AC976C2-1F45-4046-B2F8-2E288E6D2005}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{0D16B38B-CA32-4EE9-9710-6FD655BEAF8F}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{12514A7D-61D9-4C37-BC1B-162DEB9E67B2}resource=/MXP/AO0;0;WriteMethodType=U16{158DD48D-1737-41BE-B4EC-E66B4BDFE83C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{175C3D9A-F468-4331-8CAB-728A1FDA27B2}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{230B38D1-8B56-4451-855E-55F01B0EC059}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{244FAFD2-C3E0-49D6-A3E4-BD737A67ACDE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{252CD302-84F3-4E3E-92B6-5E5046DF55F1}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{2728990C-0955-4ABE-A595-1BA7CAC22012}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{27C15163-E4EE-44B0-8B99-BF3C328C3ACB}resource=/AccelerationX;0;ReadMethodType=I16{2897DBE3-A873-4026-AEA6-A38F1A626F13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{29DFD2E7-77EB-4E36-AA74-30AE988228F0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{2C9625CB-20E2-4237-9585-44C7D17B5578}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{2F22D623-CACF-4A67-9174-08942B417B0D}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{32D8B85C-4EC9-4FDB-83BB-B94383281261}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{3DD6FEA2-C30B-443B-A2DE-D511BB5E9915}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3FC14A16-BBBF-4C0B-9B72-A62A15D2A586}resource=/AccelerationZ;0;ReadMethodType=I16{48613830-9912-4349-BD50-2B27E694943B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{4CEAD8BB-FECF-4DEC-8524-E9FAA1C4B84D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{4EF86A8A-1ED2-4212-A38A-871AA0AC9DED}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{5021F2D0-2FEB-415A-A236-C91403C8C434}resource=/MXP/AO1;0;WriteMethodType=U16{553801DF-5D29-48CA-B5A5-F856F4279932}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{56318075-D11F-4659-AC03-D0751FEB1226}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{5C1B663D-FAF4-4831-BBE6-051C45990D7F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{5C3973A4-8F31-4AE4-9FAA-E0CA91B10F51}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{61A1B603-4F86-4DC9-9B6D-426B7EF107C8}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{6255D830-D90A-4488-A943-B5B3382F7B1A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{634DC4D4-541F-420E-AB3D-9984B466CBDF}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{663F166E-1631-4874-BA76-C54FBBE99E51}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{6DBFE461-A262-4359-9695-7067BBC00DBF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{71715594-4816-4D85-AB80-179F18D313CB}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{717E653E-4F6B-4A16-93C9-215929314605}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{79152894-ACE9-4CAF-B127-5C7C0A84F8A4}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{7E1B5D5C-CA6B-44B5-B7F3-E573115FE528}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{7F125321-43CF-4483-A3FC-F4DEB1914881}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{867E4EEB-23A7-4071-B170-8A90A02FEDFB}resource=/AnalogIn1;0;ReadMethodType=U16{86D88082-2D6C-47CD-9194-EDCF3A80AF61}resource=/AnalogIn3;0;ReadMethodType=U16{8810D240-93E6-4B16-BAF3-CFEEFAB8DC43}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{88A8C9B0-92CD-4926-B733-E8991C0D5DC6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{8AA6E36B-0E35-4B83-AD09-2E5987794A75}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{94B523E0-ABDB-482E-B3B6-BDC9DA454042}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{982D32E9-2FCB-4A4B-8D8B-18BE3924B5CF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{98DBF425-91B8-4CF9-8E96-EA9A7E6662F5}resource=/MXP/AI3;0;ReadMethodType=U16{995E239E-AECE-4BAD-B0DC-D9C51A83AF55}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{9F00E1FF-FAD7-4B26-81F7-0FD4CFAE6535}resource=/AnalogIn2;0;ReadMethodType=U16{A0D01AB8-BB91-4375-B91F-EFBC1E1D0A5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{A65F0CFC-BEF4-4F03-8A0A-3D6755976458}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{ADD767DC-89F6-4E99-A999-C3C511E8F049}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B186E5B0-13AF-4F67-8DCD-5CFB0E38B8CD}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{B40402A8-6BB2-410E-BE83-269307075AA4}resource=/MXP/AI1;0;ReadMethodType=U16{B59FBA04-C081-4910-979C-F475BF94B4E7}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{B6D79384-1544-48F5-8C30-015ACFAD2B7D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{BA2B9DA1-3804-44F4-B42F-B944367F4830}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB927CDD-6CEA-4FA3-BA27-F8B37B322B5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{C136E580-FDC3-4FD0-A73E-8FADF50FCEF1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{C832ECE7-8171-418B-A4A0-368DF1088B67}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{C86B5E87-7DB7-48BF-AD9F-3A17A56A19EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{CBB1C1E1-B5D6-43E3-81A4-2CE233E8F521}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{D0188368-5943-4B71-A85B-0B5C16AF1080}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{D1EBD955-AB5D-45B8-8908-322BF3D03143}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{D76A6B63-78CF-4DE7-ABC9-110AAC617555}resource=/MXP/AI0;0;ReadMethodType=U16{D9B2ADEF-4B0F-4EB1-9D0F-509EE0B9567C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{DAE46D39-D9D1-4089-8FC2-5135206CE1EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{DC25416B-85BC-4736-B363-87C87591B48D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{DF69F36E-FFFB-4674-859E-4CC1FA6F5787}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{E8D2FB32-9FA4-4993-984C-CB8CB4AC97E7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{EAF2C81F-5F74-4BC2-AE3D-33B3D5D9DD0B}resource=/AnalogIn0;0;ReadMethodType=U16{EB635694-8D5D-4B46-8BD0-5DC72292FF44}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{ED3CA370-9D51-46A4-A6BF-1D32C231286B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1AC4ABC-9F5A-430A-B8B7-ADC575BE3272}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1E9396B-8491-4765-A5F0-90494714CF93}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{F9B030D6-994B-4D02-8DD1-B94488513E67}resource=/AccelerationY;0;ReadMethodType=I16{FB0D33ED-4CE1-49A3-853E-099CFA00F792}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{FC2A8B33-1B74-4A37-A867-88C76E8C298B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{FF9337F1-C277-47DF-9AAC-D1D389DC8EA4}resource=/MXP/AI2;0;ReadMethodType=U16roboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
							<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
							<Property Name="NI.LV.FPGA.InterfaceBitfile" Type="Str">C:\Users\abs21004\Documents\GitHub\EXO1\LabView\FPGA Bitfiles\exo1_FPGATarget_EXO1_L0-+3DpafcI.lvbitx</Property>
						</Item>
						<Item Name="DataIn.vi" Type="VI" URL="../DataIn.vi">
							<Property Name="configString.guid" Type="Str">{03B4C7F9-31B1-4375-8D6C-3C498D3E98C8}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{046BD9D1-BCD3-41D3-8623-714B59353BDD}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{052DD952-8FDA-4F77-85A3-7A41612B9001}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{0AC976C2-1F45-4046-B2F8-2E288E6D2005}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{0D16B38B-CA32-4EE9-9710-6FD655BEAF8F}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{12514A7D-61D9-4C37-BC1B-162DEB9E67B2}resource=/MXP/AO0;0;WriteMethodType=U16{158DD48D-1737-41BE-B4EC-E66B4BDFE83C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{175C3D9A-F468-4331-8CAB-728A1FDA27B2}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{230B38D1-8B56-4451-855E-55F01B0EC059}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{244FAFD2-C3E0-49D6-A3E4-BD737A67ACDE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{252CD302-84F3-4E3E-92B6-5E5046DF55F1}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{2728990C-0955-4ABE-A595-1BA7CAC22012}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{27C15163-E4EE-44B0-8B99-BF3C328C3ACB}resource=/AccelerationX;0;ReadMethodType=I16{2897DBE3-A873-4026-AEA6-A38F1A626F13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{29DFD2E7-77EB-4E36-AA74-30AE988228F0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{2C9625CB-20E2-4237-9585-44C7D17B5578}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{2F22D623-CACF-4A67-9174-08942B417B0D}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{32D8B85C-4EC9-4FDB-83BB-B94383281261}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{3DD6FEA2-C30B-443B-A2DE-D511BB5E9915}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3FC14A16-BBBF-4C0B-9B72-A62A15D2A586}resource=/AccelerationZ;0;ReadMethodType=I16{48613830-9912-4349-BD50-2B27E694943B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{4CEAD8BB-FECF-4DEC-8524-E9FAA1C4B84D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{4EF86A8A-1ED2-4212-A38A-871AA0AC9DED}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{5021F2D0-2FEB-415A-A236-C91403C8C434}resource=/MXP/AO1;0;WriteMethodType=U16{553801DF-5D29-48CA-B5A5-F856F4279932}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{56318075-D11F-4659-AC03-D0751FEB1226}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{5C1B663D-FAF4-4831-BBE6-051C45990D7F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{5C3973A4-8F31-4AE4-9FAA-E0CA91B10F51}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{61A1B603-4F86-4DC9-9B6D-426B7EF107C8}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{6255D830-D90A-4488-A943-B5B3382F7B1A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{634DC4D4-541F-420E-AB3D-9984B466CBDF}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{663F166E-1631-4874-BA76-C54FBBE99E51}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{6DBFE461-A262-4359-9695-7067BBC00DBF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{71715594-4816-4D85-AB80-179F18D313CB}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{717E653E-4F6B-4A16-93C9-215929314605}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{79152894-ACE9-4CAF-B127-5C7C0A84F8A4}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{7E1B5D5C-CA6B-44B5-B7F3-E573115FE528}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{7F125321-43CF-4483-A3FC-F4DEB1914881}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{867E4EEB-23A7-4071-B170-8A90A02FEDFB}resource=/AnalogIn1;0;ReadMethodType=U16{86D88082-2D6C-47CD-9194-EDCF3A80AF61}resource=/AnalogIn3;0;ReadMethodType=U16{8810D240-93E6-4B16-BAF3-CFEEFAB8DC43}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{88A8C9B0-92CD-4926-B733-E8991C0D5DC6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{8AA6E36B-0E35-4B83-AD09-2E5987794A75}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{94B523E0-ABDB-482E-B3B6-BDC9DA454042}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{982D32E9-2FCB-4A4B-8D8B-18BE3924B5CF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{98DBF425-91B8-4CF9-8E96-EA9A7E6662F5}resource=/MXP/AI3;0;ReadMethodType=U16{995E239E-AECE-4BAD-B0DC-D9C51A83AF55}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{9F00E1FF-FAD7-4B26-81F7-0FD4CFAE6535}resource=/AnalogIn2;0;ReadMethodType=U16{A0D01AB8-BB91-4375-B91F-EFBC1E1D0A5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{A65F0CFC-BEF4-4F03-8A0A-3D6755976458}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{ADD767DC-89F6-4E99-A999-C3C511E8F049}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B186E5B0-13AF-4F67-8DCD-5CFB0E38B8CD}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{B40402A8-6BB2-410E-BE83-269307075AA4}resource=/MXP/AI1;0;ReadMethodType=U16{B59FBA04-C081-4910-979C-F475BF94B4E7}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{B6D79384-1544-48F5-8C30-015ACFAD2B7D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{BA2B9DA1-3804-44F4-B42F-B944367F4830}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB927CDD-6CEA-4FA3-BA27-F8B37B322B5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{C136E580-FDC3-4FD0-A73E-8FADF50FCEF1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{C832ECE7-8171-418B-A4A0-368DF1088B67}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{C86B5E87-7DB7-48BF-AD9F-3A17A56A19EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{CBB1C1E1-B5D6-43E3-81A4-2CE233E8F521}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{D0188368-5943-4B71-A85B-0B5C16AF1080}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{D1EBD955-AB5D-45B8-8908-322BF3D03143}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{D76A6B63-78CF-4DE7-ABC9-110AAC617555}resource=/MXP/AI0;0;ReadMethodType=U16{D9B2ADEF-4B0F-4EB1-9D0F-509EE0B9567C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{DAE46D39-D9D1-4089-8FC2-5135206CE1EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{DC25416B-85BC-4736-B363-87C87591B48D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{DF69F36E-FFFB-4674-859E-4CC1FA6F5787}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{E8D2FB32-9FA4-4993-984C-CB8CB4AC97E7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{EAF2C81F-5F74-4BC2-AE3D-33B3D5D9DD0B}resource=/AnalogIn0;0;ReadMethodType=U16{EB635694-8D5D-4B46-8BD0-5DC72292FF44}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{ED3CA370-9D51-46A4-A6BF-1D32C231286B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1AC4ABC-9F5A-430A-B8B7-ADC575BE3272}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1E9396B-8491-4765-A5F0-90494714CF93}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{F9B030D6-994B-4D02-8DD1-B94488513E67}resource=/AccelerationY;0;ReadMethodType=I16{FB0D33ED-4CE1-49A3-853E-099CFA00F792}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{FC2A8B33-1B74-4A37-A867-88C76E8C298B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{FF9337F1-C277-47DF-9AAC-D1D389DC8EA4}resource=/MXP/AI2;0;ReadMethodType=U16roboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
							<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
							<Property Name="NI.LV.FPGA.InterfaceBitfile" Type="Str">C:\Users\abs21004\Documents\GitHub\EXO1\LabView\FPGA Bitfiles\exo1_FPGATarget_DataIn_R1wp+qdUXdo.lvbitx</Property>
						</Item>
						<Item Name="4xFFT.vi" Type="VI" URL="../4xFFT.vi">
							<Property Name="configString.guid" Type="Str">{03B4C7F9-31B1-4375-8D6C-3C498D3E98C8}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{046BD9D1-BCD3-41D3-8623-714B59353BDD}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{052DD952-8FDA-4F77-85A3-7A41612B9001}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{0AC976C2-1F45-4046-B2F8-2E288E6D2005}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{0D16B38B-CA32-4EE9-9710-6FD655BEAF8F}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{12514A7D-61D9-4C37-BC1B-162DEB9E67B2}resource=/MXP/AO0;0;WriteMethodType=U16{158DD48D-1737-41BE-B4EC-E66B4BDFE83C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{175C3D9A-F468-4331-8CAB-728A1FDA27B2}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{230B38D1-8B56-4451-855E-55F01B0EC059}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{244FAFD2-C3E0-49D6-A3E4-BD737A67ACDE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{252CD302-84F3-4E3E-92B6-5E5046DF55F1}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{2728990C-0955-4ABE-A595-1BA7CAC22012}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{27C15163-E4EE-44B0-8B99-BF3C328C3ACB}resource=/AccelerationX;0;ReadMethodType=I16{2897DBE3-A873-4026-AEA6-A38F1A626F13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{29DFD2E7-77EB-4E36-AA74-30AE988228F0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{2C9625CB-20E2-4237-9585-44C7D17B5578}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{2F22D623-CACF-4A67-9174-08942B417B0D}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{32D8B85C-4EC9-4FDB-83BB-B94383281261}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{3DD6FEA2-C30B-443B-A2DE-D511BB5E9915}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3FC14A16-BBBF-4C0B-9B72-A62A15D2A586}resource=/AccelerationZ;0;ReadMethodType=I16{48613830-9912-4349-BD50-2B27E694943B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{4CEAD8BB-FECF-4DEC-8524-E9FAA1C4B84D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{4EF86A8A-1ED2-4212-A38A-871AA0AC9DED}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{5021F2D0-2FEB-415A-A236-C91403C8C434}resource=/MXP/AO1;0;WriteMethodType=U16{553801DF-5D29-48CA-B5A5-F856F4279932}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{56318075-D11F-4659-AC03-D0751FEB1226}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{5C1B663D-FAF4-4831-BBE6-051C45990D7F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{5C3973A4-8F31-4AE4-9FAA-E0CA91B10F51}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{61A1B603-4F86-4DC9-9B6D-426B7EF107C8}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{6255D830-D90A-4488-A943-B5B3382F7B1A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{634DC4D4-541F-420E-AB3D-9984B466CBDF}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{663F166E-1631-4874-BA76-C54FBBE99E51}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{6DBFE461-A262-4359-9695-7067BBC00DBF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{71715594-4816-4D85-AB80-179F18D313CB}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{717E653E-4F6B-4A16-93C9-215929314605}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{79152894-ACE9-4CAF-B127-5C7C0A84F8A4}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{7E1B5D5C-CA6B-44B5-B7F3-E573115FE528}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{7F125321-43CF-4483-A3FC-F4DEB1914881}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{867E4EEB-23A7-4071-B170-8A90A02FEDFB}resource=/AnalogIn1;0;ReadMethodType=U16{86D88082-2D6C-47CD-9194-EDCF3A80AF61}resource=/AnalogIn3;0;ReadMethodType=U16{8810D240-93E6-4B16-BAF3-CFEEFAB8DC43}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{88A8C9B0-92CD-4926-B733-E8991C0D5DC6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{8AA6E36B-0E35-4B83-AD09-2E5987794A75}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{94B523E0-ABDB-482E-B3B6-BDC9DA454042}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{982D32E9-2FCB-4A4B-8D8B-18BE3924B5CF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{98DBF425-91B8-4CF9-8E96-EA9A7E6662F5}resource=/MXP/AI3;0;ReadMethodType=U16{995E239E-AECE-4BAD-B0DC-D9C51A83AF55}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{9F00E1FF-FAD7-4B26-81F7-0FD4CFAE6535}resource=/AnalogIn2;0;ReadMethodType=U16{A0D01AB8-BB91-4375-B91F-EFBC1E1D0A5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{A65F0CFC-BEF4-4F03-8A0A-3D6755976458}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{ADD767DC-89F6-4E99-A999-C3C511E8F049}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B186E5B0-13AF-4F67-8DCD-5CFB0E38B8CD}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{B40402A8-6BB2-410E-BE83-269307075AA4}resource=/MXP/AI1;0;ReadMethodType=U16{B59FBA04-C081-4910-979C-F475BF94B4E7}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{B6D79384-1544-48F5-8C30-015ACFAD2B7D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{BA2B9DA1-3804-44F4-B42F-B944367F4830}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB927CDD-6CEA-4FA3-BA27-F8B37B322B5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{C136E580-FDC3-4FD0-A73E-8FADF50FCEF1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{C832ECE7-8171-418B-A4A0-368DF1088B67}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{C86B5E87-7DB7-48BF-AD9F-3A17A56A19EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{CBB1C1E1-B5D6-43E3-81A4-2CE233E8F521}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{D0188368-5943-4B71-A85B-0B5C16AF1080}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{D1EBD955-AB5D-45B8-8908-322BF3D03143}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{D76A6B63-78CF-4DE7-ABC9-110AAC617555}resource=/MXP/AI0;0;ReadMethodType=U16{D9B2ADEF-4B0F-4EB1-9D0F-509EE0B9567C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{DAE46D39-D9D1-4089-8FC2-5135206CE1EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{DC25416B-85BC-4736-B363-87C87591B48D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{DF69F36E-FFFB-4674-859E-4CC1FA6F5787}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{E8D2FB32-9FA4-4993-984C-CB8CB4AC97E7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{EAF2C81F-5F74-4BC2-AE3D-33B3D5D9DD0B}resource=/AnalogIn0;0;ReadMethodType=U16{EB635694-8D5D-4B46-8BD0-5DC72292FF44}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{ED3CA370-9D51-46A4-A6BF-1D32C231286B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1AC4ABC-9F5A-430A-B8B7-ADC575BE3272}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1E9396B-8491-4765-A5F0-90494714CF93}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{F9B030D6-994B-4D02-8DD1-B94488513E67}resource=/AccelerationY;0;ReadMethodType=I16{FB0D33ED-4CE1-49A3-853E-099CFA00F792}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{FC2A8B33-1B74-4A37-A867-88C76E8C298B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{FF9337F1-C277-47DF-9AAC-D1D389DC8EA4}resource=/MXP/AI2;0;ReadMethodType=U16roboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
							<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
						</Item>
						<Item Name="RNN.vi" Type="VI" URL="../RNN.vi">
							<Property Name="configString.guid" Type="Str">{03B4C7F9-31B1-4375-8D6C-3C498D3E98C8}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{046BD9D1-BCD3-41D3-8623-714B59353BDD}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{052DD952-8FDA-4F77-85A3-7A41612B9001}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{0AC976C2-1F45-4046-B2F8-2E288E6D2005}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{0D16B38B-CA32-4EE9-9710-6FD655BEAF8F}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{12514A7D-61D9-4C37-BC1B-162DEB9E67B2}resource=/MXP/AO0;0;WriteMethodType=U16{158DD48D-1737-41BE-B4EC-E66B4BDFE83C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{175C3D9A-F468-4331-8CAB-728A1FDA27B2}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{230B38D1-8B56-4451-855E-55F01B0EC059}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{244FAFD2-C3E0-49D6-A3E4-BD737A67ACDE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{252CD302-84F3-4E3E-92B6-5E5046DF55F1}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{2728990C-0955-4ABE-A595-1BA7CAC22012}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{27C15163-E4EE-44B0-8B99-BF3C328C3ACB}resource=/AccelerationX;0;ReadMethodType=I16{2897DBE3-A873-4026-AEA6-A38F1A626F13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{29DFD2E7-77EB-4E36-AA74-30AE988228F0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{2C9625CB-20E2-4237-9585-44C7D17B5578}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{2F22D623-CACF-4A67-9174-08942B417B0D}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{32D8B85C-4EC9-4FDB-83BB-B94383281261}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{3DD6FEA2-C30B-443B-A2DE-D511BB5E9915}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3FC14A16-BBBF-4C0B-9B72-A62A15D2A586}resource=/AccelerationZ;0;ReadMethodType=I16{48613830-9912-4349-BD50-2B27E694943B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{4CEAD8BB-FECF-4DEC-8524-E9FAA1C4B84D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{4EF86A8A-1ED2-4212-A38A-871AA0AC9DED}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{5021F2D0-2FEB-415A-A236-C91403C8C434}resource=/MXP/AO1;0;WriteMethodType=U16{553801DF-5D29-48CA-B5A5-F856F4279932}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{56318075-D11F-4659-AC03-D0751FEB1226}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{5C1B663D-FAF4-4831-BBE6-051C45990D7F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{5C3973A4-8F31-4AE4-9FAA-E0CA91B10F51}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{61A1B603-4F86-4DC9-9B6D-426B7EF107C8}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{6255D830-D90A-4488-A943-B5B3382F7B1A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{634DC4D4-541F-420E-AB3D-9984B466CBDF}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{663F166E-1631-4874-BA76-C54FBBE99E51}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{6DBFE461-A262-4359-9695-7067BBC00DBF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{71715594-4816-4D85-AB80-179F18D313CB}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{717E653E-4F6B-4A16-93C9-215929314605}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{79152894-ACE9-4CAF-B127-5C7C0A84F8A4}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{7E1B5D5C-CA6B-44B5-B7F3-E573115FE528}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{7F125321-43CF-4483-A3FC-F4DEB1914881}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{867E4EEB-23A7-4071-B170-8A90A02FEDFB}resource=/AnalogIn1;0;ReadMethodType=U16{86D88082-2D6C-47CD-9194-EDCF3A80AF61}resource=/AnalogIn3;0;ReadMethodType=U16{8810D240-93E6-4B16-BAF3-CFEEFAB8DC43}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{88A8C9B0-92CD-4926-B733-E8991C0D5DC6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{8AA6E36B-0E35-4B83-AD09-2E5987794A75}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{94B523E0-ABDB-482E-B3B6-BDC9DA454042}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{982D32E9-2FCB-4A4B-8D8B-18BE3924B5CF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{98DBF425-91B8-4CF9-8E96-EA9A7E6662F5}resource=/MXP/AI3;0;ReadMethodType=U16{995E239E-AECE-4BAD-B0DC-D9C51A83AF55}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{9F00E1FF-FAD7-4B26-81F7-0FD4CFAE6535}resource=/AnalogIn2;0;ReadMethodType=U16{A0D01AB8-BB91-4375-B91F-EFBC1E1D0A5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{A65F0CFC-BEF4-4F03-8A0A-3D6755976458}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{ADD767DC-89F6-4E99-A999-C3C511E8F049}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B186E5B0-13AF-4F67-8DCD-5CFB0E38B8CD}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{B40402A8-6BB2-410E-BE83-269307075AA4}resource=/MXP/AI1;0;ReadMethodType=U16{B59FBA04-C081-4910-979C-F475BF94B4E7}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{B6D79384-1544-48F5-8C30-015ACFAD2B7D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{BA2B9DA1-3804-44F4-B42F-B944367F4830}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB927CDD-6CEA-4FA3-BA27-F8B37B322B5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{C136E580-FDC3-4FD0-A73E-8FADF50FCEF1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{C832ECE7-8171-418B-A4A0-368DF1088B67}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{C86B5E87-7DB7-48BF-AD9F-3A17A56A19EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{CBB1C1E1-B5D6-43E3-81A4-2CE233E8F521}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{D0188368-5943-4B71-A85B-0B5C16AF1080}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{D1EBD955-AB5D-45B8-8908-322BF3D03143}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{D76A6B63-78CF-4DE7-ABC9-110AAC617555}resource=/MXP/AI0;0;ReadMethodType=U16{D9B2ADEF-4B0F-4EB1-9D0F-509EE0B9567C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{DAE46D39-D9D1-4089-8FC2-5135206CE1EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{DC25416B-85BC-4736-B363-87C87591B48D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{DF69F36E-FFFB-4674-859E-4CC1FA6F5787}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{E8D2FB32-9FA4-4993-984C-CB8CB4AC97E7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{EAF2C81F-5F74-4BC2-AE3D-33B3D5D9DD0B}resource=/AnalogIn0;0;ReadMethodType=U16{EB635694-8D5D-4B46-8BD0-5DC72292FF44}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{ED3CA370-9D51-46A4-A6BF-1D32C231286B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1AC4ABC-9F5A-430A-B8B7-ADC575BE3272}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1E9396B-8491-4765-A5F0-90494714CF93}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{F9B030D6-994B-4D02-8DD1-B94488513E67}resource=/AccelerationY;0;ReadMethodType=I16{FB0D33ED-4CE1-49A3-853E-099CFA00F792}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{FC2A8B33-1B74-4A37-A867-88C76E8C298B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{FF9337F1-C277-47DF-9AAC-D1D389DC8EA4}resource=/MXP/AI2;0;ReadMethodType=U16roboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
							<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
						</Item>
						<Item Name="RNN-Node.vi" Type="VI" URL="../RNN-Node.vi">
							<Property Name="configString.guid" Type="Str">{03B4C7F9-31B1-4375-8D6C-3C498D3E98C8}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{046BD9D1-BCD3-41D3-8623-714B59353BDD}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{052DD952-8FDA-4F77-85A3-7A41612B9001}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{0AC976C2-1F45-4046-B2F8-2E288E6D2005}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{0D16B38B-CA32-4EE9-9710-6FD655BEAF8F}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{12514A7D-61D9-4C37-BC1B-162DEB9E67B2}resource=/MXP/AO0;0;WriteMethodType=U16{158DD48D-1737-41BE-B4EC-E66B4BDFE83C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{175C3D9A-F468-4331-8CAB-728A1FDA27B2}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{230B38D1-8B56-4451-855E-55F01B0EC059}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{244FAFD2-C3E0-49D6-A3E4-BD737A67ACDE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{252CD302-84F3-4E3E-92B6-5E5046DF55F1}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{2728990C-0955-4ABE-A595-1BA7CAC22012}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{27C15163-E4EE-44B0-8B99-BF3C328C3ACB}resource=/AccelerationX;0;ReadMethodType=I16{2897DBE3-A873-4026-AEA6-A38F1A626F13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{29DFD2E7-77EB-4E36-AA74-30AE988228F0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{2C9625CB-20E2-4237-9585-44C7D17B5578}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{2F22D623-CACF-4A67-9174-08942B417B0D}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{32D8B85C-4EC9-4FDB-83BB-B94383281261}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{3DD6FEA2-C30B-443B-A2DE-D511BB5E9915}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3FC14A16-BBBF-4C0B-9B72-A62A15D2A586}resource=/AccelerationZ;0;ReadMethodType=I16{48613830-9912-4349-BD50-2B27E694943B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{4CEAD8BB-FECF-4DEC-8524-E9FAA1C4B84D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{4EF86A8A-1ED2-4212-A38A-871AA0AC9DED}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{5021F2D0-2FEB-415A-A236-C91403C8C434}resource=/MXP/AO1;0;WriteMethodType=U16{553801DF-5D29-48CA-B5A5-F856F4279932}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{56318075-D11F-4659-AC03-D0751FEB1226}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{5C1B663D-FAF4-4831-BBE6-051C45990D7F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{5C3973A4-8F31-4AE4-9FAA-E0CA91B10F51}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{61A1B603-4F86-4DC9-9B6D-426B7EF107C8}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{6255D830-D90A-4488-A943-B5B3382F7B1A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{634DC4D4-541F-420E-AB3D-9984B466CBDF}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{663F166E-1631-4874-BA76-C54FBBE99E51}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{6DBFE461-A262-4359-9695-7067BBC00DBF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{71715594-4816-4D85-AB80-179F18D313CB}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{717E653E-4F6B-4A16-93C9-215929314605}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{79152894-ACE9-4CAF-B127-5C7C0A84F8A4}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{7E1B5D5C-CA6B-44B5-B7F3-E573115FE528}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{7F125321-43CF-4483-A3FC-F4DEB1914881}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{867E4EEB-23A7-4071-B170-8A90A02FEDFB}resource=/AnalogIn1;0;ReadMethodType=U16{86D88082-2D6C-47CD-9194-EDCF3A80AF61}resource=/AnalogIn3;0;ReadMethodType=U16{8810D240-93E6-4B16-BAF3-CFEEFAB8DC43}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{88A8C9B0-92CD-4926-B733-E8991C0D5DC6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{8AA6E36B-0E35-4B83-AD09-2E5987794A75}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{94B523E0-ABDB-482E-B3B6-BDC9DA454042}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{982D32E9-2FCB-4A4B-8D8B-18BE3924B5CF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{98DBF425-91B8-4CF9-8E96-EA9A7E6662F5}resource=/MXP/AI3;0;ReadMethodType=U16{995E239E-AECE-4BAD-B0DC-D9C51A83AF55}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{9F00E1FF-FAD7-4B26-81F7-0FD4CFAE6535}resource=/AnalogIn2;0;ReadMethodType=U16{A0D01AB8-BB91-4375-B91F-EFBC1E1D0A5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{A65F0CFC-BEF4-4F03-8A0A-3D6755976458}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{ADD767DC-89F6-4E99-A999-C3C511E8F049}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B186E5B0-13AF-4F67-8DCD-5CFB0E38B8CD}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{B40402A8-6BB2-410E-BE83-269307075AA4}resource=/MXP/AI1;0;ReadMethodType=U16{B59FBA04-C081-4910-979C-F475BF94B4E7}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{B6D79384-1544-48F5-8C30-015ACFAD2B7D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{BA2B9DA1-3804-44F4-B42F-B944367F4830}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB927CDD-6CEA-4FA3-BA27-F8B37B322B5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{C136E580-FDC3-4FD0-A73E-8FADF50FCEF1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{C832ECE7-8171-418B-A4A0-368DF1088B67}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{C86B5E87-7DB7-48BF-AD9F-3A17A56A19EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{CBB1C1E1-B5D6-43E3-81A4-2CE233E8F521}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{D0188368-5943-4B71-A85B-0B5C16AF1080}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{D1EBD955-AB5D-45B8-8908-322BF3D03143}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{D76A6B63-78CF-4DE7-ABC9-110AAC617555}resource=/MXP/AI0;0;ReadMethodType=U16{D9B2ADEF-4B0F-4EB1-9D0F-509EE0B9567C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{DAE46D39-D9D1-4089-8FC2-5135206CE1EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{DC25416B-85BC-4736-B363-87C87591B48D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{DF69F36E-FFFB-4674-859E-4CC1FA6F5787}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{E8D2FB32-9FA4-4993-984C-CB8CB4AC97E7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{EAF2C81F-5F74-4BC2-AE3D-33B3D5D9DD0B}resource=/AnalogIn0;0;ReadMethodType=U16{EB635694-8D5D-4B46-8BD0-5DC72292FF44}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{ED3CA370-9D51-46A4-A6BF-1D32C231286B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1AC4ABC-9F5A-430A-B8B7-ADC575BE3272}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1E9396B-8491-4765-A5F0-90494714CF93}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{F9B030D6-994B-4D02-8DD1-B94488513E67}resource=/AccelerationY;0;ReadMethodType=I16{FB0D33ED-4CE1-49A3-853E-099CFA00F792}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{FC2A8B33-1B74-4A37-A867-88C76E8C298B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{FF9337F1-C277-47DF-9AAC-D1D389DC8EA4}resource=/MXP/AI2;0;ReadMethodType=U16roboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
							<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
						</Item>
						<Item Name="RNN-PreProcess-Node.vi" Type="VI" URL="../RNN-PreProcess-Node.vi">
							<Property Name="configString.guid" Type="Str">{03B4C7F9-31B1-4375-8D6C-3C498D3E98C8}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{046BD9D1-BCD3-41D3-8623-714B59353BDD}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{052DD952-8FDA-4F77-85A3-7A41612B9001}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{0AC976C2-1F45-4046-B2F8-2E288E6D2005}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{0D16B38B-CA32-4EE9-9710-6FD655BEAF8F}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{12514A7D-61D9-4C37-BC1B-162DEB9E67B2}resource=/MXP/AO0;0;WriteMethodType=U16{158DD48D-1737-41BE-B4EC-E66B4BDFE83C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{175C3D9A-F468-4331-8CAB-728A1FDA27B2}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{230B38D1-8B56-4451-855E-55F01B0EC059}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{244FAFD2-C3E0-49D6-A3E4-BD737A67ACDE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{252CD302-84F3-4E3E-92B6-5E5046DF55F1}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{2728990C-0955-4ABE-A595-1BA7CAC22012}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{27C15163-E4EE-44B0-8B99-BF3C328C3ACB}resource=/AccelerationX;0;ReadMethodType=I16{2897DBE3-A873-4026-AEA6-A38F1A626F13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{29DFD2E7-77EB-4E36-AA74-30AE988228F0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{2C9625CB-20E2-4237-9585-44C7D17B5578}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{2F22D623-CACF-4A67-9174-08942B417B0D}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{32D8B85C-4EC9-4FDB-83BB-B94383281261}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{3DD6FEA2-C30B-443B-A2DE-D511BB5E9915}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3FC14A16-BBBF-4C0B-9B72-A62A15D2A586}resource=/AccelerationZ;0;ReadMethodType=I16{48613830-9912-4349-BD50-2B27E694943B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{4CEAD8BB-FECF-4DEC-8524-E9FAA1C4B84D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{4EF86A8A-1ED2-4212-A38A-871AA0AC9DED}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{5021F2D0-2FEB-415A-A236-C91403C8C434}resource=/MXP/AO1;0;WriteMethodType=U16{553801DF-5D29-48CA-B5A5-F856F4279932}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{56318075-D11F-4659-AC03-D0751FEB1226}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{5C1B663D-FAF4-4831-BBE6-051C45990D7F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{5C3973A4-8F31-4AE4-9FAA-E0CA91B10F51}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{61A1B603-4F86-4DC9-9B6D-426B7EF107C8}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{6255D830-D90A-4488-A943-B5B3382F7B1A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{634DC4D4-541F-420E-AB3D-9984B466CBDF}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{663F166E-1631-4874-BA76-C54FBBE99E51}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{6DBFE461-A262-4359-9695-7067BBC00DBF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{71715594-4816-4D85-AB80-179F18D313CB}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{717E653E-4F6B-4A16-93C9-215929314605}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{79152894-ACE9-4CAF-B127-5C7C0A84F8A4}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{7E1B5D5C-CA6B-44B5-B7F3-E573115FE528}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{7F125321-43CF-4483-A3FC-F4DEB1914881}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{867E4EEB-23A7-4071-B170-8A90A02FEDFB}resource=/AnalogIn1;0;ReadMethodType=U16{86D88082-2D6C-47CD-9194-EDCF3A80AF61}resource=/AnalogIn3;0;ReadMethodType=U16{8810D240-93E6-4B16-BAF3-CFEEFAB8DC43}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{88A8C9B0-92CD-4926-B733-E8991C0D5DC6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{8AA6E36B-0E35-4B83-AD09-2E5987794A75}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{94B523E0-ABDB-482E-B3B6-BDC9DA454042}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{982D32E9-2FCB-4A4B-8D8B-18BE3924B5CF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{98DBF425-91B8-4CF9-8E96-EA9A7E6662F5}resource=/MXP/AI3;0;ReadMethodType=U16{995E239E-AECE-4BAD-B0DC-D9C51A83AF55}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{9F00E1FF-FAD7-4B26-81F7-0FD4CFAE6535}resource=/AnalogIn2;0;ReadMethodType=U16{A0D01AB8-BB91-4375-B91F-EFBC1E1D0A5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{A65F0CFC-BEF4-4F03-8A0A-3D6755976458}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{ADD767DC-89F6-4E99-A999-C3C511E8F049}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B186E5B0-13AF-4F67-8DCD-5CFB0E38B8CD}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{B40402A8-6BB2-410E-BE83-269307075AA4}resource=/MXP/AI1;0;ReadMethodType=U16{B59FBA04-C081-4910-979C-F475BF94B4E7}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{B6D79384-1544-48F5-8C30-015ACFAD2B7D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{BA2B9DA1-3804-44F4-B42F-B944367F4830}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB927CDD-6CEA-4FA3-BA27-F8B37B322B5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{C136E580-FDC3-4FD0-A73E-8FADF50FCEF1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{C832ECE7-8171-418B-A4A0-368DF1088B67}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{C86B5E87-7DB7-48BF-AD9F-3A17A56A19EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{CBB1C1E1-B5D6-43E3-81A4-2CE233E8F521}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{D0188368-5943-4B71-A85B-0B5C16AF1080}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{D1EBD955-AB5D-45B8-8908-322BF3D03143}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{D76A6B63-78CF-4DE7-ABC9-110AAC617555}resource=/MXP/AI0;0;ReadMethodType=U16{D9B2ADEF-4B0F-4EB1-9D0F-509EE0B9567C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{DAE46D39-D9D1-4089-8FC2-5135206CE1EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{DC25416B-85BC-4736-B363-87C87591B48D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{DF69F36E-FFFB-4674-859E-4CC1FA6F5787}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{E8D2FB32-9FA4-4993-984C-CB8CB4AC97E7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{EAF2C81F-5F74-4BC2-AE3D-33B3D5D9DD0B}resource=/AnalogIn0;0;ReadMethodType=U16{EB635694-8D5D-4B46-8BD0-5DC72292FF44}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{ED3CA370-9D51-46A4-A6BF-1D32C231286B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1AC4ABC-9F5A-430A-B8B7-ADC575BE3272}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1E9396B-8491-4765-A5F0-90494714CF93}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{F9B030D6-994B-4D02-8DD1-B94488513E67}resource=/AccelerationY;0;ReadMethodType=I16{FB0D33ED-4CE1-49A3-853E-099CFA00F792}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{FC2A8B33-1B74-4A37-A867-88C76E8C298B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{FF9337F1-C277-47DF-9AAC-D1D389DC8EA4}resource=/MXP/AI2;0;ReadMethodType=U16roboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
							<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
						</Item>
						<Item Name="Motor-Controls.vi" Type="VI" URL="../Motor-Controls.vi">
							<Property Name="configString.guid" Type="Str">{03B4C7F9-31B1-4375-8D6C-3C498D3E98C8}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{046BD9D1-BCD3-41D3-8623-714B59353BDD}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{052DD952-8FDA-4F77-85A3-7A41612B9001}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{0AC976C2-1F45-4046-B2F8-2E288E6D2005}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{0D16B38B-CA32-4EE9-9710-6FD655BEAF8F}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{12514A7D-61D9-4C37-BC1B-162DEB9E67B2}resource=/MXP/AO0;0;WriteMethodType=U16{158DD48D-1737-41BE-B4EC-E66B4BDFE83C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{175C3D9A-F468-4331-8CAB-728A1FDA27B2}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{230B38D1-8B56-4451-855E-55F01B0EC059}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{244FAFD2-C3E0-49D6-A3E4-BD737A67ACDE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{252CD302-84F3-4E3E-92B6-5E5046DF55F1}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{2728990C-0955-4ABE-A595-1BA7CAC22012}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{27C15163-E4EE-44B0-8B99-BF3C328C3ACB}resource=/AccelerationX;0;ReadMethodType=I16{2897DBE3-A873-4026-AEA6-A38F1A626F13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{29DFD2E7-77EB-4E36-AA74-30AE988228F0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{2C9625CB-20E2-4237-9585-44C7D17B5578}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{2F22D623-CACF-4A67-9174-08942B417B0D}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{32D8B85C-4EC9-4FDB-83BB-B94383281261}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{3DD6FEA2-C30B-443B-A2DE-D511BB5E9915}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3FC14A16-BBBF-4C0B-9B72-A62A15D2A586}resource=/AccelerationZ;0;ReadMethodType=I16{48613830-9912-4349-BD50-2B27E694943B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{4CEAD8BB-FECF-4DEC-8524-E9FAA1C4B84D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{4EF86A8A-1ED2-4212-A38A-871AA0AC9DED}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{5021F2D0-2FEB-415A-A236-C91403C8C434}resource=/MXP/AO1;0;WriteMethodType=U16{553801DF-5D29-48CA-B5A5-F856F4279932}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{56318075-D11F-4659-AC03-D0751FEB1226}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{5C1B663D-FAF4-4831-BBE6-051C45990D7F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{5C3973A4-8F31-4AE4-9FAA-E0CA91B10F51}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{61A1B603-4F86-4DC9-9B6D-426B7EF107C8}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{6255D830-D90A-4488-A943-B5B3382F7B1A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{634DC4D4-541F-420E-AB3D-9984B466CBDF}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{663F166E-1631-4874-BA76-C54FBBE99E51}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{6DBFE461-A262-4359-9695-7067BBC00DBF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{71715594-4816-4D85-AB80-179F18D313CB}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{717E653E-4F6B-4A16-93C9-215929314605}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{79152894-ACE9-4CAF-B127-5C7C0A84F8A4}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{7E1B5D5C-CA6B-44B5-B7F3-E573115FE528}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{7F125321-43CF-4483-A3FC-F4DEB1914881}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{867E4EEB-23A7-4071-B170-8A90A02FEDFB}resource=/AnalogIn1;0;ReadMethodType=U16{86D88082-2D6C-47CD-9194-EDCF3A80AF61}resource=/AnalogIn3;0;ReadMethodType=U16{8810D240-93E6-4B16-BAF3-CFEEFAB8DC43}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{88A8C9B0-92CD-4926-B733-E8991C0D5DC6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{8AA6E36B-0E35-4B83-AD09-2E5987794A75}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{94B523E0-ABDB-482E-B3B6-BDC9DA454042}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{982D32E9-2FCB-4A4B-8D8B-18BE3924B5CF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{98DBF425-91B8-4CF9-8E96-EA9A7E6662F5}resource=/MXP/AI3;0;ReadMethodType=U16{995E239E-AECE-4BAD-B0DC-D9C51A83AF55}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{9F00E1FF-FAD7-4B26-81F7-0FD4CFAE6535}resource=/AnalogIn2;0;ReadMethodType=U16{A0D01AB8-BB91-4375-B91F-EFBC1E1D0A5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{A65F0CFC-BEF4-4F03-8A0A-3D6755976458}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{ADD767DC-89F6-4E99-A999-C3C511E8F049}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B186E5B0-13AF-4F67-8DCD-5CFB0E38B8CD}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{B40402A8-6BB2-410E-BE83-269307075AA4}resource=/MXP/AI1;0;ReadMethodType=U16{B59FBA04-C081-4910-979C-F475BF94B4E7}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{B6D79384-1544-48F5-8C30-015ACFAD2B7D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{BA2B9DA1-3804-44F4-B42F-B944367F4830}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB927CDD-6CEA-4FA3-BA27-F8B37B322B5A}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{C136E580-FDC3-4FD0-A73E-8FADF50FCEF1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{C832ECE7-8171-418B-A4A0-368DF1088B67}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{C86B5E87-7DB7-48BF-AD9F-3A17A56A19EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{CBB1C1E1-B5D6-43E3-81A4-2CE233E8F521}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{D0188368-5943-4B71-A85B-0B5C16AF1080}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{D1EBD955-AB5D-45B8-8908-322BF3D03143}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{D76A6B63-78CF-4DE7-ABC9-110AAC617555}resource=/MXP/AI0;0;ReadMethodType=U16{D9B2ADEF-4B0F-4EB1-9D0F-509EE0B9567C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{DAE46D39-D9D1-4089-8FC2-5135206CE1EB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{DC25416B-85BC-4736-B363-87C87591B48D}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{DF69F36E-FFFB-4674-859E-4CC1FA6F5787}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{E8D2FB32-9FA4-4993-984C-CB8CB4AC97E7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{EAF2C81F-5F74-4BC2-AE3D-33B3D5D9DD0B}resource=/AnalogIn0;0;ReadMethodType=U16{EB635694-8D5D-4B46-8BD0-5DC72292FF44}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{ED3CA370-9D51-46A4-A6BF-1D32C231286B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1AC4ABC-9F5A-430A-B8B7-ADC575BE3272}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{F1E9396B-8491-4765-A5F0-90494714CF93}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{F9B030D6-994B-4D02-8DD1-B94488513E67}resource=/AccelerationY;0;ReadMethodType=I16{FB0D33ED-4CE1-49A3-853E-099CFA00F792}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{FC2A8B33-1B74-4A37-A867-88C76E8C298B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{FF9337F1-C277-47DF-9AAC-D1D389DC8EA4}resource=/MXP/AI2;0;ReadMethodType=U16roboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
							<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
							<Property Name="NI.LV.FPGA.InterfaceBitfile" Type="Str">C:\Users\abs21004\Documents\GitHub\EXO1\LabView\FPGA Bitfiles\exo1_FPGATarget_Motor-Controls_I39uQKWCuG0.lvbitx</Property>
						</Item>
					</Item>
					<Item Name="40 MHz Onboard Clock" Type="FPGA Base Clock">
						<Property Name="FPGA.PersistentID" Type="Str">{A65F0CFC-BEF4-4F03-8A0A-3D6755976458}</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig" Type="Str">ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig.Accuracy" Type="Dbl">100</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig.ClockSignalName" Type="Str">Clk40</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig.MaxDutyCycle" Type="Dbl">50</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig.MaxFrequency" Type="Dbl">40000000</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig.MinDutyCycle" Type="Dbl">50</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig.MinFrequency" Type="Dbl">40000000</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig.NominalFrequency" Type="Dbl">40000000</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig.PeakPeriodJitter" Type="Dbl">250</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig.ResourceName" Type="Str">40 MHz Onboard Clock</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig.SupportAndRequireRuntimeEnableDisable" Type="Bool">false</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig.TopSignalConnect" Type="Str">Clk40</Property>
						<Property Name="NI.LV.FPGA.BaseTSConfig.VariableFrequency" Type="Bool">false</Property>
						<Property Name="NI.LV.FPGA.Valid" Type="Bool">true</Property>
						<Property Name="NI.LV.FPGA.Version" Type="Int">5</Property>
					</Item>
					<Item Name="Dependencies" Type="Dependencies">
						<Item Name="vi.lib" Type="Folder">
							<Item Name="LVFixedPointQuantizationPolicyTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/fxp/LVFixedPointQuantizationPolicyTypeDef.ctl"/>
							<Item Name="lvSimController.dll" Type="Document" URL="/&lt;vilib&gt;/rvi/Simulation/lvSimController.dll"/>
						</Item>
						<Item Name="user.lib" Type="Folder">
							<Item Name="NI I2C IP FPGA.lvlib" Type="Library" URL="/&lt;userlib&gt;/_NI I2C IP/FPGA/NI I2C IP FPGA.lvlib"/>
						</Item>
					</Item>
					<Item Name="Build Specifications" Type="Build">
						<Item Name="DataIn" Type="{F4C5E96F-7410-48A5-BB87-3559BC9B167F}">
							<Property Name="AllowEnableRemoval" Type="Bool">false</Property>
							<Property Name="BuildSpecDecription" Type="Str"></Property>
							<Property Name="BuildSpecName" Type="Str">DataIn</Property>
							<Property Name="Comp.BitfileName" Type="Str">untitledproject1_FPGATarget_DataIn_AqlDSnv3Cuc.lvbitx</Property>
							<Property Name="Comp.CustomXilinxParameters" Type="Str"></Property>
							<Property Name="Comp.MaxFanout" Type="Int">-1</Property>
							<Property Name="Comp.RandomSeed" Type="Bool">false</Property>
							<Property Name="Comp.Version.Build" Type="Int">0</Property>
							<Property Name="Comp.Version.Fix" Type="Int">0</Property>
							<Property Name="Comp.Version.Major" Type="Int">1</Property>
							<Property Name="Comp.Version.Minor" Type="Int">0</Property>
							<Property Name="Comp.VersionAutoIncrement" Type="Bool">false</Property>
							<Property Name="Comp.Vivado.EnableMultiThreading" Type="Bool">true</Property>
							<Property Name="Comp.Vivado.OptDirective" Type="Str"></Property>
							<Property Name="Comp.Vivado.PhysOptDirective" Type="Str"></Property>
							<Property Name="Comp.Vivado.PlaceDirective" Type="Str"></Property>
							<Property Name="Comp.Vivado.RouteDirective" Type="Str"></Property>
							<Property Name="Comp.Vivado.RunPowerOpt" Type="Bool">false</Property>
							<Property Name="Comp.Vivado.Strategy" Type="Str">Default</Property>
							<Property Name="Comp.Xilinx.DesignStrategy" Type="Str">balanced</Property>
							<Property Name="Comp.Xilinx.MapEffort" Type="Str">default(noTiming)</Property>
							<Property Name="Comp.Xilinx.ParEffort" Type="Str">standard</Property>
							<Property Name="Comp.Xilinx.SynthEffort" Type="Str">normal</Property>
							<Property Name="Comp.Xilinx.SynthGoal" Type="Str">speed</Property>
							<Property Name="Comp.Xilinx.UseRecommended" Type="Bool">true</Property>
							<Property Name="DefaultBuildSpec" Type="Bool">true</Property>
							<Property Name="DestinationDirectory" Type="Path">FPGA Bitfiles</Property>
							<Property Name="NI.LV.FPGA.LastCompiledBitfilePath" Type="Path">/C/Users/abs21004/Documents/GitHub/EXO1/LabView/FPGA Bitfiles/exo1_FPGATarget_DataIn_R1wp+qdUXdo.lvbitx</Property>
							<Property Name="NI.LV.FPGA.LastCompiledBitfilePathRelativeToProject" Type="Path">FPGA Bitfiles/exo1_FPGATarget_DataIn_R1wp+qdUXdo.lvbitx</Property>
							<Property Name="ProjectPath" Type="Path">/C/Users/FKN19004/Documents/Untitled Project 1.lvproj</Property>
							<Property Name="RelativePath" Type="Bool">true</Property>
							<Property Name="RunWhenLoaded" Type="Bool">false</Property>
							<Property Name="SupportDownload" Type="Bool">true</Property>
							<Property Name="SupportResourceEstimation" Type="Bool">false</Property>
							<Property Name="TargetName" Type="Str">FPGA Target</Property>
							<Property Name="TopLevelVI" Type="Ref">/New Folder/roboRIO-330-FRC/Chassis/FPGA Target/EXO1_VI/DataIn.vi</Property>
						</Item>
						<Item Name="EXO1" Type="{F4C5E96F-7410-48A5-BB87-3559BC9B167F}">
							<Property Name="AllowEnableRemoval" Type="Bool">false</Property>
							<Property Name="BuildSpecDecription" Type="Str"></Property>
							<Property Name="BuildSpecName" Type="Str">EXO1</Property>
							<Property Name="Comp.BitfileName" Type="Str">exo1_FPGATarget_EXO1_L0-+3DpafcI.lvbitx</Property>
							<Property Name="Comp.CustomXilinxParameters" Type="Str"></Property>
							<Property Name="Comp.MaxFanout" Type="Int">-1</Property>
							<Property Name="Comp.RandomSeed" Type="Bool">false</Property>
							<Property Name="Comp.Version.Build" Type="Int">0</Property>
							<Property Name="Comp.Version.Fix" Type="Int">0</Property>
							<Property Name="Comp.Version.Major" Type="Int">1</Property>
							<Property Name="Comp.Version.Minor" Type="Int">0</Property>
							<Property Name="Comp.VersionAutoIncrement" Type="Bool">false</Property>
							<Property Name="Comp.Vivado.EnableMultiThreading" Type="Bool">true</Property>
							<Property Name="Comp.Vivado.OptDirective" Type="Str"></Property>
							<Property Name="Comp.Vivado.PhysOptDirective" Type="Str"></Property>
							<Property Name="Comp.Vivado.PlaceDirective" Type="Str"></Property>
							<Property Name="Comp.Vivado.RouteDirective" Type="Str"></Property>
							<Property Name="Comp.Vivado.RunPowerOpt" Type="Bool">false</Property>
							<Property Name="Comp.Vivado.Strategy" Type="Str">Default</Property>
							<Property Name="Comp.Xilinx.DesignStrategy" Type="Str">balanced</Property>
							<Property Name="Comp.Xilinx.MapEffort" Type="Str">default(noTiming)</Property>
							<Property Name="Comp.Xilinx.ParEffort" Type="Str">standard</Property>
							<Property Name="Comp.Xilinx.SynthEffort" Type="Str">normal</Property>
							<Property Name="Comp.Xilinx.SynthGoal" Type="Str">speed</Property>
							<Property Name="Comp.Xilinx.UseRecommended" Type="Bool">true</Property>
							<Property Name="DefaultBuildSpec" Type="Bool">true</Property>
							<Property Name="DestinationDirectory" Type="Path">FPGA Bitfiles</Property>
							<Property Name="NI.LV.FPGA.LastCompiledBitfilePath" Type="Path">/C/Users/abs21004/Documents/GitHub/EXO1/LabView/FPGA Bitfiles/exo1_FPGATarget_EXO1_L0-+3DpafcI.lvbitx</Property>
							<Property Name="NI.LV.FPGA.LastCompiledBitfilePathRelativeToProject" Type="Path">FPGA Bitfiles/exo1_FPGATarget_EXO1_L0-+3DpafcI.lvbitx</Property>
							<Property Name="ProjectPath" Type="Path">/C/Users/abs21004/Documents/GitHub/EXO1/LabView/EXO1.lvproj</Property>
							<Property Name="RelativePath" Type="Bool">true</Property>
							<Property Name="RunWhenLoaded" Type="Bool">false</Property>
							<Property Name="SupportDownload" Type="Bool">true</Property>
							<Property Name="SupportResourceEstimation" Type="Bool">false</Property>
							<Property Name="TargetName" Type="Str">FPGA Target</Property>
							<Property Name="TopLevelVI" Type="Ref">/New Folder/roboRIO-330-FRC/Chassis/FPGA Target/EXO1_VI/EXO1.vi</Property>
						</Item>
						<Item Name="Motor-Controls" Type="{F4C5E96F-7410-48A5-BB87-3559BC9B167F}">
							<Property Name="AllowEnableRemoval" Type="Bool">false</Property>
							<Property Name="BuildSpecDecription" Type="Str"></Property>
							<Property Name="BuildSpecName" Type="Str">Motor-Controls</Property>
							<Property Name="Comp.BitfileName" Type="Str">exo1_FPGATarget_Motor-Controls_I39uQKWCuG0.lvbitx</Property>
							<Property Name="Comp.CustomXilinxParameters" Type="Str"></Property>
							<Property Name="Comp.MaxFanout" Type="Int">-1</Property>
							<Property Name="Comp.RandomSeed" Type="Bool">false</Property>
							<Property Name="Comp.Version.Build" Type="Int">0</Property>
							<Property Name="Comp.Version.Fix" Type="Int">0</Property>
							<Property Name="Comp.Version.Major" Type="Int">1</Property>
							<Property Name="Comp.Version.Minor" Type="Int">0</Property>
							<Property Name="Comp.VersionAutoIncrement" Type="Bool">false</Property>
							<Property Name="Comp.Vivado.EnableMultiThreading" Type="Bool">true</Property>
							<Property Name="Comp.Vivado.OptDirective" Type="Str"></Property>
							<Property Name="Comp.Vivado.PhysOptDirective" Type="Str"></Property>
							<Property Name="Comp.Vivado.PlaceDirective" Type="Str"></Property>
							<Property Name="Comp.Vivado.RouteDirective" Type="Str"></Property>
							<Property Name="Comp.Vivado.RunPowerOpt" Type="Bool">false</Property>
							<Property Name="Comp.Vivado.Strategy" Type="Str">Default</Property>
							<Property Name="Comp.Xilinx.DesignStrategy" Type="Str">balanced</Property>
							<Property Name="Comp.Xilinx.MapEffort" Type="Str">default(noTiming)</Property>
							<Property Name="Comp.Xilinx.ParEffort" Type="Str">standard</Property>
							<Property Name="Comp.Xilinx.SynthEffort" Type="Str">normal</Property>
							<Property Name="Comp.Xilinx.SynthGoal" Type="Str">speed</Property>
							<Property Name="Comp.Xilinx.UseRecommended" Type="Bool">true</Property>
							<Property Name="DefaultBuildSpec" Type="Bool">true</Property>
							<Property Name="DestinationDirectory" Type="Path">FPGA Bitfiles</Property>
							<Property Name="NI.LV.FPGA.LastCompiledBitfilePath" Type="Path">/C/Users/abs21004/Documents/GitHub/EXO1/LabView/FPGA Bitfiles/exo1_FPGATarget_Motor-Controls_I39uQKWCuG0.lvbitx</Property>
							<Property Name="NI.LV.FPGA.LastCompiledBitfilePathRelativeToProject" Type="Path">FPGA Bitfiles/exo1_FPGATarget_Motor-Controls_I39uQKWCuG0.lvbitx</Property>
							<Property Name="ProjectPath" Type="Path">/C/Users/abs21004/Documents/GitHub/EXO1/LabView/EXO1.lvproj</Property>
							<Property Name="RelativePath" Type="Bool">true</Property>
							<Property Name="RunWhenLoaded" Type="Bool">false</Property>
							<Property Name="SupportDownload" Type="Bool">true</Property>
							<Property Name="SupportResourceEstimation" Type="Bool">false</Property>
							<Property Name="TargetName" Type="Str">FPGA Target</Property>
							<Property Name="TopLevelVI" Type="Ref">/New Folder/roboRIO-330-FRC/Chassis/FPGA Target/EXO1_VI/Motor-Controls.vi</Property>
						</Item>
					</Item>
				</Item>
			</Item>
			<Item Name="RT-RNN.vi" Type="VI" URL="../RT-RNN.vi"/>
			<Item Name="RT-Motor-Controls.vi" Type="VI" URL="../RT-Motor-Controls.vi"/>
			<Item Name="RT-Data-in.vi" Type="VI" URL="../RT-Data-in.vi"/>
			<Item Name="RT-LSTM.vi" Type="VI" URL="../RT-LSTM.vi"/>
			<Item Name="RT-sigmoid.vi" Type="VI" URL="../RT-sigmoid.vi"/>
			<Item Name="RT-Tanh.vi" Type="VI" URL="../RT-Tanh.vi"/>
			<Item Name="RT-LTSM-Layer.vi" Type="VI" URL="../RT-LTSM-Layer.vi"/>
			<Item Name="RT-Init.vi" Type="VI" URL="../RT-Init.vi"/>
			<Item Name="RT-Terminate.vi" Type="VI" URL="../RT-Terminate.vi"/>
			<Item Name="Dependencies" Type="Dependencies">
				<Item Name="vi.lib" Type="Folder">
					<Item Name="CANStatusToError.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANStatusToError.vi"/>
					<Item Name="CANCreateInterface.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANCreateInterface.vi"/>
					<Item Name="CANFrameControl.ctl" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANFrameControl.ctl"/>
					<Item Name="whitespace.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/whitespace.ctl"/>
					<Item Name="Trim Whitespace.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Trim Whitespace.vi"/>
					<Item Name="CANStart.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANStart.vi"/>
					<Item Name="CANRead.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANRead.vi"/>
					<Item Name="CANCloseInterface.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANCloseInterface.vi"/>
					<Item Name="CANWrite.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANWrite.vi"/>
					<Item Name="Simple Error Handler.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Simple Error Handler.vi"/>
					<Item Name="General Error Handler.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/General Error Handler.vi"/>
					<Item Name="General Error Handler Core CORE.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/General Error Handler Core CORE.vi"/>
					<Item Name="Error Code Database.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Code Database.vi"/>
					<Item Name="Check Special Tags.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Check Special Tags.vi"/>
					<Item Name="TagReturnType.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/TagReturnType.ctl"/>
					<Item Name="DialogTypeEnum.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/DialogTypeEnum.ctl"/>
					<Item Name="Not Found Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Not Found Dialog.vi"/>
					<Item Name="BuildHelpPath.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/BuildHelpPath.vi"/>
					<Item Name="GetHelpDir.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/GetHelpDir.vi"/>
					<Item Name="Three Button Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog.vi"/>
					<Item Name="Three Button Dialog CORE.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog CORE.vi"/>
					<Item Name="Longest Line Length in Pixels.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Longest Line Length in Pixels.vi"/>
					<Item Name="Get Text Rect.vi" Type="VI" URL="/&lt;vilib&gt;/picture/picture.llb/Get Text Rect.vi"/>
					<Item Name="Convert property node font to graphics font.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Convert property node font to graphics font.vi"/>
					<Item Name="Get String Text Bounds.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Get String Text Bounds.vi"/>
					<Item Name="LVBoundsTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVBoundsTypeDef.ctl"/>
					<Item Name="LVRectTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVRectTypeDef.ctl"/>
					<Item Name="Format Message String.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Format Message String.vi"/>
					<Item Name="Find Tag.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Find Tag.vi"/>
					<Item Name="Search and Replace Pattern.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Search and Replace Pattern.vi"/>
					<Item Name="Set Bold Text.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set Bold Text.vi"/>
					<Item Name="Details Display Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Details Display Dialog.vi"/>
					<Item Name="eventvkey.ctl" Type="VI" URL="/&lt;vilib&gt;/event_ctls.llb/eventvkey.ctl"/>
					<Item Name="Clear Errors.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Clear Errors.vi"/>
					<Item Name="ErrWarn.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/ErrWarn.ctl"/>
					<Item Name="Set String Value.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set String Value.vi"/>
					<Item Name="GetRTHostConnectedProp.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/GetRTHostConnectedProp.vi"/>
					<Item Name="DialogType.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/DialogType.ctl"/>
					<Item Name="CANFilterMode.ctl" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANFilterMode.ctl"/>
					<Item Name="CANFilterSetting.ctl" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANFilterSetting.ctl"/>
					<Item Name="CANSetFilterMode.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANSetFilterMode.vi"/>
					<Item Name="CANSetFilterValue.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANSetFilterValue.vi"/>
					<Item Name="CANStop.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANStop.vi"/>
					<Item Name="CANWait.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANWait.vi"/>
					<Item Name="CANWaitTransmitComplete.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANWaitTransmitComplete.vi"/>
					<Item Name="CANWaitInterfaceCommunicating.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANWaitInterfaceCommunicating.vi"/>
					<Item Name="CANWaitRemoteWakeup.vi" Type="VI" URL="/&lt;vilib&gt;/RioEmbeddedCAN/RioEmbeddedCAN.llb/CANWaitRemoteWakeup.vi"/>
					<Item Name="Error Cluster From Error Code.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Cluster From Error Code.vi"/>
				</Item>
				<Item Name="rioembeddedcanlvapi.dll" Type="Document" URL="rioembeddedcanlvapi.dll">
					<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
				</Item>
				<Item Name="NiFpgaLv.dll" Type="Document" URL="NiFpgaLv.dll">
					<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
				</Item>
				<Item Name="RT-RIOtoVolt.vi" Type="VI" URL="../RT-RIOtoVolt.vi"/>
				<Item Name="RT-ANN-Node.vi" Type="VI" URL="../RT-ANN-Node.vi"/>
				<Item Name="RT-Mode-Selection.vi" Type="VI" URL="../RT-Mode-Selection.vi"/>
			</Item>
			<Item Name="Build Specifications" Type="Build"/>
		</Item>
	</Item>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="Dependencies" Type="Dependencies"/>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
