<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="20008000">
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
		<Item Name="RoboRIO" Type="Folder">
			<Item Name="FPGA Target" Type="FPGA Target">
				<Property Name="AutoRun" Type="Bool">false</Property>
				<Property Name="configString.guid" Type="Str">{036DF3CB-3458-47DA-90D2-ACEBD7625A99}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{04A71BF2-AA06-4E1D-A1F1-4CCE89F7B91E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{06FD7152-9008-47B3-8F28-696D3C97C875}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{0A28A788-8D2F-4F77-B8F4-B02412C9E51D}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{0DFF9EEA-2F3A-40A7-B72E-7160789DC629}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{109438B6-C5C8-460D-8842-0C730A9DB172}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{16180A68-244E-4ABC-882E-1FE9B2B5A96C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{296720F7-2657-4D43-A0A9-060C6F1F71B7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{2A5A8526-2503-4447-A9D2-D0F9BB7F931B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{2CBA67DC-957B-4396-8C20-436E7DD8E037}resource=/AccelerationX;0;ReadMethodType=I16{2EA5272D-1487-41D3-BC4B-1436CDBE0846}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{30579913-32F8-4529-86E4-2BB0188209FE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{30C62109-A65A-415B-A8AA-34215F05DCA1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{30D2C326-733D-4A43-95E8-11C81988D3FC}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{3370460E-8621-42F8-90B0-212D3317FD61}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{362EC810-C30D-44C2-A76B-45D223D88E60}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{37044D48-48FC-4F11-808B-EF94C581D0F5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{378FE4D5-976C-49EC-839F-6CBA8EFEC038}resource=/AnalogIn1;0;ReadMethodType=U16{38760416-0184-4B83-B111-585BFAA5A42C}resource=/AnalogIn2;0;ReadMethodType=U16{3AF87085-C2D1-41BC-A2A9-56CDB792EC80}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3B381BAA-6D28-49B6-8DC5-15FDD5C4BF49}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{3C8D7016-5E0D-4EE7-A1AC-287832A90691}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{3DC9B286-C7F9-45E1-8A13-2E2ABAB62F02}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{3EFB491E-CC90-4707-97F1-CEF3377DE51E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{438FBFE9-1175-4302-8799-ABE13FADC1E9}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{459CD608-BCEF-4C73-A04E-2F6C62DA6B70}resource=/MXP/AI3;0;ReadMethodType=U16{46600E50-6CB1-473D-B999-CB5E5F92C1B3}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{4912ABC2-A372-4177-BA59-CF0D7E0A66C5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{51D565AA-225D-4DE8-B2BB-BB4D09B6344F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{525B2E61-0035-4646-A684-F06840DAD30F}resource=/MXP/AO1;0;WriteMethodType=U16{5E523F57-FC5A-4CB0-B6B7-53410322A70B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{6299C9A5-9E6C-4202-8AA1-6C9234EBC8CC}resource=/MXP/AI0;0;ReadMethodType=U16{6311591E-2529-4E29-A5EC-E25215AA2800}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{647515CF-4FD6-44AE-940A-593D8B0F3F54}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{65FC489C-2D76-48D1-B84D-8088BAA3E1FE}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{662271A6-E297-43C1-B67B-8CBEC0308530}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{66D48B2E-EF8B-487B-986E-B9AB947B0A2C}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{6C85F277-497E-42F5-A56A-4D23C4FABC76}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{7285CC5C-BF99-4FA4-9684-DFA7CE8FD910}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{72DCE64E-9177-492B-BD12-80E1EEBEF5A0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{73D8FDCB-00D6-4147-AD76-5D8D0ECC6F64}resource=/MXP/AI1;0;ReadMethodType=U16{778C9CFE-6CCB-4AF1-AF7C-AAEB85FDEB84}resource=/AnalogIn3;0;ReadMethodType=U16{7F8F077D-C47A-4410-BE5E-B5CE9B625C96}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{83F1A921-DD61-43B5-B66A-0E1E0C19D92E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{83FBA6E6-1D0E-42E1-A48E-ECCD892EA9DF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{86E46AFA-83C7-4773-94F6-DDF1CFDA8D13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{87BFE617-D994-4FB1-AD0B-06D368560E1B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{8AA92FF1-2E57-4E8F-A1AF-E76166C7A929}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{8C0967AE-99BE-4CCB-8E41-CF5B82FA0448}resource=/AccelerationY;0;ReadMethodType=I16{8D67E739-75D7-408C-9DA2-344640F6B590}resource=/MXP/AO0;0;WriteMethodType=U16{8FAE59E8-FC01-4BD5-93D4-B2129CEB25A9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{905E4E04-9B97-4455-826C-358C303A66D9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{94BC5F9E-84CE-4335-B757-979F4963B020}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{9728004D-D9C4-4B4F-A67E-CC26BA4EBCEF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{97CAF5F2-DC18-4632-9E68-808A91D820FC}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{9A784D1F-1C3F-4189-B170-9FD46FFCAD4F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{9C3CDA6D-737A-4224-9FD8-A0ADE35D2391}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{9C67454B-2364-4CDD-B7E0-16E4CD604A12}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{9E8DD9B6-C50C-49E2-A4E4-9D083DE1D83D}resource=/AnalogIn0;0;ReadMethodType=U16{A3D0E05D-9F7B-465A-B624-11E7DADD99C5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{A84CE67F-F21D-4207-992B-891D9CFA7E4A}resource=/AccelerationZ;0;ReadMethodType=I16{AC8D2DAE-59CB-4752-8421-E8EABF034750}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B3CCC653-1E16-4D31-A352-43B3B99AB2B5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{B74D8E39-1031-4249-A233-2313A21CF8B6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB9A8698-091A-487B-894F-C8F9A096902C}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{BBE1574D-BCFB-45D7-B9C6-034ABA8CDA19}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{C1FD649E-ED1B-4636-9FA1-81E2424A0B5B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{C65D4E96-3027-4822-BBE4-8672ECE90161}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{C7F5AB61-2A34-470A-B642-31F2C9D4D4CA}resource=/MXP/AI2;0;ReadMethodType=U16{DEA72476-F7DD-47DB-B9F6-C38A618AC0B9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{DF7B2D72-6610-43C2-B410-502A228D3D55}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{DF9C85A7-54C3-4A96-89E4-6963493CC050}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{E3DB5321-3865-4D01-BB41-8E05EEEE615B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{E7C586F7-C290-4CD8-993F-4B8487E6A2BB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{F0D73D2A-511D-4CE1-8A15-83E49BA6EF94}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{F57A09EE-EA36-4D3E-87AD-CC4FF3BED1A5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{F677DA21-0D2D-446C-8002-2905D115F9F2}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{FA805930-C536-4C9C-8E24-5951CF15D0D4}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
				<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
				<Property Name="Mode" Type="Int">0</Property>
				<Property Name="NI.LV.FPGA.CLIPDeclarationsArraySize" Type="Int">0</Property>
				<Property Name="NI.LV.FPGA.CLIPDeclarationSet" Type="Xml">
<CLIPDeclarationSet>
</CLIPDeclarationSet></Property>
				<Property Name="NI.LV.FPGA.CompileConfigString" Type="Str">roboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
				<Property Name="NI.LV.FPGA.Version" Type="Int">6</Property>
				<Property Name="niFpga_TopLevelVIID" Type="Path">/C/Users/abs21004/Downloads/4xFFT.vi</Property>
				<Property Name="Resource Name" Type="Str">172.22.11.2</Property>
				<Property Name="SWEmulationSubMode" Type="UInt">0</Property>
				<Property Name="SWEmulationVIPath" Type="Path"></Property>
				<Property Name="Target Class" Type="Str">roboRIO</Property>
				<Property Name="Top-Level Timing Source" Type="Str">40 MHz Onboard Clock</Property>
				<Property Name="Top-Level Timing Source Is Default" Type="Bool">true</Property>
				<Item Name="Analog" Type="Folder">
					<Item Name="AnalogIn0" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AnalogIn0</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{9E8DD9B6-C50C-49E2-A4E4-9D083DE1D83D}</Property>
					</Item>
					<Item Name="AnalogIn1" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AnalogIn1</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{378FE4D5-976C-49EC-839F-6CBA8EFEC038}</Property>
					</Item>
					<Item Name="AnalogIn2" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AnalogIn2</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{38760416-0184-4B83-B111-585BFAA5A42C}</Property>
					</Item>
					<Item Name="AnalogIn3" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AnalogIn3</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{778C9CFE-6CCB-4AF1-AF7C-AAEB85FDEB84}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{9C3CDA6D-737A-4224-9FD8-A0ADE35D2391}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{7285CC5C-BF99-4FA4-9684-DFA7CE8FD910}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{905E4E04-9B97-4455-826C-358C303A66D9}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{C1FD649E-ED1B-4636-9FA1-81E2424A0B5B}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{B3CCC653-1E16-4D31-A352-43B3B99AB2B5}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{E3DB5321-3865-4D01-BB41-8E05EEEE615B}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{4912ABC2-A372-4177-BA59-CF0D7E0A66C5}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{6311591E-2529-4E29-A5EC-E25215AA2800}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{72DCE64E-9177-492B-BD12-80E1EEBEF5A0}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{30C62109-A65A-415B-A8AA-34215F05DCA1}</Property>
					</Item>
				</Item>
				<Item Name="I2C" Type="Folder">
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
						<Property Name="FPGA.PersistentID" Type="Str">{662271A6-E297-43C1-B67B-8CBEC0308530}</Property>
					</Item>
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
						<Property Name="FPGA.PersistentID" Type="Str">{036DF3CB-3458-47DA-90D2-ACEBD7625A99}</Property>
					</Item>
				</Item>
				<Item Name="MXP" Type="Folder">
					<Item Name="Analog" Type="Folder">
						<Item Name="MXP/AI0" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/MXP/AI0</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{6299C9A5-9E6C-4202-8AA1-6C9234EBC8CC}</Property>
						</Item>
						<Item Name="MXP/AI1" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/MXP/AI1</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{73D8FDCB-00D6-4147-AD76-5D8D0ECC6F64}</Property>
						</Item>
						<Item Name="MXP/AI2" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/MXP/AI2</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{C7F5AB61-2A34-470A-B642-31F2C9D4D4CA}</Property>
						</Item>
						<Item Name="MXP/AI3" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/MXP/AI3</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{459CD608-BCEF-4C73-A04E-2F6C62DA6B70}</Property>
						</Item>
						<Item Name="MXP/AO0" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/MXP/AO0</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{8D67E739-75D7-408C-9DA2-344640F6B590}</Property>
						</Item>
						<Item Name="MXP/AO1" Type="Elemental IO">
							<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/MXP/AO1</Value>
   </Attribute>
</AttributeSet>
</Property>
							<Property Name="FPGA.PersistentID" Type="Str">{525B2E61-0035-4646-A684-F06840DAD30F}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{109438B6-C5C8-460D-8842-0C730A9DB172}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{3B381BAA-6D28-49B6-8DC5-15FDD5C4BF49}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{C65D4E96-3027-4822-BBE4-8672ECE90161}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{9C67454B-2364-4CDD-B7E0-16E4CD604A12}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{97CAF5F2-DC18-4632-9E68-808A91D820FC}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{B74D8E39-1031-4249-A233-2313A21CF8B6}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{DF7B2D72-6610-43C2-B410-502A228D3D55}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{FA805930-C536-4C9C-8E24-5951CF15D0D4}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{DEA72476-F7DD-47DB-B9F6-C38A618AC0B9}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{2EA5272D-1487-41D3-BC4B-1436CDBE0846}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{AC8D2DAE-59CB-4752-8421-E8EABF034750}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{3AF87085-C2D1-41BC-A2A9-56CDB792EC80}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{86E46AFA-83C7-4773-94F6-DDF1CFDA8D13}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{30D2C326-733D-4A43-95E8-11C81988D3FC}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{2A5A8526-2503-4447-A9D2-D0F9BB7F931B}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{BBE1574D-BCFB-45D7-B9C6-034ABA8CDA19}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{362EC810-C30D-44C2-A76B-45D223D88E60}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{46600E50-6CB1-473D-B999-CB5E5F92C1B3}</Property>
						</Item>
					</Item>
				</Item>
				<Item Name="Onboard I/O" Type="Folder">
					<Item Name="LED" Type="Folder">
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
							<Property Name="FPGA.PersistentID" Type="Str">{7F8F077D-C47A-4410-BE5E-B5CE9B625C96}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{F57A09EE-EA36-4D3E-87AD-CC4FF3BED1A5}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{F0D73D2A-511D-4CE1-8A15-83E49BA6EF94}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{A3D0E05D-9F7B-465A-B624-11E7DADD99C5}</Property>
						</Item>
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
							<Property Name="FPGA.PersistentID" Type="Str">{647515CF-4FD6-44AE-940A-593D8B0F3F54}</Property>
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
							<Property Name="FPGA.PersistentID" Type="Str">{6C85F277-497E-42F5-A56A-4D23C4FABC76}</Property>
						</Item>
					</Item>
					<Item Name="AccelerationX" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AccelerationX</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{2CBA67DC-957B-4396-8C20-436E7DD8E037}</Property>
					</Item>
					<Item Name="AccelerationY" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AccelerationY</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{8C0967AE-99BE-4CCB-8E41-CF5B82FA0448}</Property>
					</Item>
					<Item Name="AccelerationZ" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/AccelerationZ</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{A84CE67F-F21D-4207-992B-891D9CFA7E4A}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{83F1A921-DD61-43B5-B66A-0E1E0C19D92E}</Property>
					</Item>
					<Item Name="System Reset" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/System Reset</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{0A28A788-8D2F-4F77-B8F4-B02412C9E51D}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{65FC489C-2D76-48D1-B84D-8088BAA3E1FE}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{438FBFE9-1175-4302-8799-ABE13FADC1E9}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{3EFB491E-CC90-4707-97F1-CEF3377DE51E}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{83FBA6E6-1D0E-42E1-A48E-ECCD892EA9DF}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{DF9C85A7-54C3-4A96-89E4-6963493CC050}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{9A784D1F-1C3F-4189-B170-9FD46FFCAD4F}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{5E523F57-FC5A-4CB0-B6B7-53410322A70B}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{8AA92FF1-2E57-4E8F-A1AF-E76166C7A929}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{0DFF9EEA-2F3A-40A7-B72E-7160789DC629}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{37044D48-48FC-4F11-808B-EF94C581D0F5}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{51D565AA-225D-4DE8-B2BB-BB4D09B6344F}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{66D48B2E-EF8B-487B-986E-B9AB947B0A2C}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{296720F7-2657-4D43-A0A9-060C6F1F71B7}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{E7C586F7-C290-4CD8-993F-4B8487E6A2BB}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{04A71BF2-AA06-4E1D-A1F1-4CCE89F7B91E}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{3C8D7016-5E0D-4EE7-A1AC-287832A90691}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{9728004D-D9C4-4B4F-A67E-CC26BA4EBCEF}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{BB9A8698-091A-487B-894F-C8F9A096902C}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{87BFE617-D994-4FB1-AD0B-06D368560E1B}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{8FAE59E8-FC01-4BD5-93D4-B2129CEB25A9}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{3DC9B286-C7F9-45E1-8A13-2E2ABAB62F02}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{3370460E-8621-42F8-90B0-212D3317FD61}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{30579913-32F8-4529-86E4-2BB0188209FE}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{16180A68-244E-4ABC-882E-1FE9B2B5A96C}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{F677DA21-0D2D-446C-8002-2905D115F9F2}</Property>
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
						<Property Name="FPGA.PersistentID" Type="Str">{94BC5F9E-84CE-4335-B757-979F4963B020}</Property>
					</Item>
				</Item>
				<Item Name="4xFFT.vi" Type="VI" URL="../4xFFT.vi">
					<Property Name="configString.guid" Type="Str">{036DF3CB-3458-47DA-90D2-ACEBD7625A99}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{04A71BF2-AA06-4E1D-A1F1-4CCE89F7B91E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{06FD7152-9008-47B3-8F28-696D3C97C875}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{0A28A788-8D2F-4F77-B8F4-B02412C9E51D}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{0DFF9EEA-2F3A-40A7-B72E-7160789DC629}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{109438B6-C5C8-460D-8842-0C730A9DB172}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{16180A68-244E-4ABC-882E-1FE9B2B5A96C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{296720F7-2657-4D43-A0A9-060C6F1F71B7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{2A5A8526-2503-4447-A9D2-D0F9BB7F931B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{2CBA67DC-957B-4396-8C20-436E7DD8E037}resource=/AccelerationX;0;ReadMethodType=I16{2EA5272D-1487-41D3-BC4B-1436CDBE0846}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{30579913-32F8-4529-86E4-2BB0188209FE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{30C62109-A65A-415B-A8AA-34215F05DCA1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{30D2C326-733D-4A43-95E8-11C81988D3FC}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{3370460E-8621-42F8-90B0-212D3317FD61}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{362EC810-C30D-44C2-A76B-45D223D88E60}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{37044D48-48FC-4F11-808B-EF94C581D0F5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{378FE4D5-976C-49EC-839F-6CBA8EFEC038}resource=/AnalogIn1;0;ReadMethodType=U16{38760416-0184-4B83-B111-585BFAA5A42C}resource=/AnalogIn2;0;ReadMethodType=U16{3AF87085-C2D1-41BC-A2A9-56CDB792EC80}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3B381BAA-6D28-49B6-8DC5-15FDD5C4BF49}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{3C8D7016-5E0D-4EE7-A1AC-287832A90691}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{3DC9B286-C7F9-45E1-8A13-2E2ABAB62F02}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{3EFB491E-CC90-4707-97F1-CEF3377DE51E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{438FBFE9-1175-4302-8799-ABE13FADC1E9}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{459CD608-BCEF-4C73-A04E-2F6C62DA6B70}resource=/MXP/AI3;0;ReadMethodType=U16{46600E50-6CB1-473D-B999-CB5E5F92C1B3}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{4912ABC2-A372-4177-BA59-CF0D7E0A66C5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{51D565AA-225D-4DE8-B2BB-BB4D09B6344F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{525B2E61-0035-4646-A684-F06840DAD30F}resource=/MXP/AO1;0;WriteMethodType=U16{5E523F57-FC5A-4CB0-B6B7-53410322A70B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{6299C9A5-9E6C-4202-8AA1-6C9234EBC8CC}resource=/MXP/AI0;0;ReadMethodType=U16{6311591E-2529-4E29-A5EC-E25215AA2800}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{647515CF-4FD6-44AE-940A-593D8B0F3F54}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{65FC489C-2D76-48D1-B84D-8088BAA3E1FE}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{662271A6-E297-43C1-B67B-8CBEC0308530}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{66D48B2E-EF8B-487B-986E-B9AB947B0A2C}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{6C85F277-497E-42F5-A56A-4D23C4FABC76}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{7285CC5C-BF99-4FA4-9684-DFA7CE8FD910}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{72DCE64E-9177-492B-BD12-80E1EEBEF5A0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{73D8FDCB-00D6-4147-AD76-5D8D0ECC6F64}resource=/MXP/AI1;0;ReadMethodType=U16{778C9CFE-6CCB-4AF1-AF7C-AAEB85FDEB84}resource=/AnalogIn3;0;ReadMethodType=U16{7F8F077D-C47A-4410-BE5E-B5CE9B625C96}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{83F1A921-DD61-43B5-B66A-0E1E0C19D92E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{83FBA6E6-1D0E-42E1-A48E-ECCD892EA9DF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{86E46AFA-83C7-4773-94F6-DDF1CFDA8D13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{87BFE617-D994-4FB1-AD0B-06D368560E1B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{8AA92FF1-2E57-4E8F-A1AF-E76166C7A929}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{8C0967AE-99BE-4CCB-8E41-CF5B82FA0448}resource=/AccelerationY;0;ReadMethodType=I16{8D67E739-75D7-408C-9DA2-344640F6B590}resource=/MXP/AO0;0;WriteMethodType=U16{8FAE59E8-FC01-4BD5-93D4-B2129CEB25A9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{905E4E04-9B97-4455-826C-358C303A66D9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{94BC5F9E-84CE-4335-B757-979F4963B020}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{9728004D-D9C4-4B4F-A67E-CC26BA4EBCEF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{97CAF5F2-DC18-4632-9E68-808A91D820FC}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{9A784D1F-1C3F-4189-B170-9FD46FFCAD4F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{9C3CDA6D-737A-4224-9FD8-A0ADE35D2391}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{9C67454B-2364-4CDD-B7E0-16E4CD604A12}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{9E8DD9B6-C50C-49E2-A4E4-9D083DE1D83D}resource=/AnalogIn0;0;ReadMethodType=U16{A3D0E05D-9F7B-465A-B624-11E7DADD99C5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{A84CE67F-F21D-4207-992B-891D9CFA7E4A}resource=/AccelerationZ;0;ReadMethodType=I16{AC8D2DAE-59CB-4752-8421-E8EABF034750}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B3CCC653-1E16-4D31-A352-43B3B99AB2B5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{B74D8E39-1031-4249-A233-2313A21CF8B6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB9A8698-091A-487B-894F-C8F9A096902C}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{BBE1574D-BCFB-45D7-B9C6-034ABA8CDA19}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{C1FD649E-ED1B-4636-9FA1-81E2424A0B5B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{C65D4E96-3027-4822-BBE4-8672ECE90161}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{C7F5AB61-2A34-470A-B642-31F2C9D4D4CA}resource=/MXP/AI2;0;ReadMethodType=U16{DEA72476-F7DD-47DB-B9F6-C38A618AC0B9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{DF7B2D72-6610-43C2-B410-502A228D3D55}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{DF9C85A7-54C3-4A96-89E4-6963493CC050}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{E3DB5321-3865-4D01-BB41-8E05EEEE615B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{E7C586F7-C290-4CD8-993F-4B8487E6A2BB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{F0D73D2A-511D-4CE1-8A15-83E49BA6EF94}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{F57A09EE-EA36-4D3E-87AD-CC4FF3BED1A5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{F677DA21-0D2D-446C-8002-2905D115F9F2}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{FA805930-C536-4C9C-8E24-5951CF15D0D4}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
					<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
					<Property Name="NI.LV.FPGA.InterfaceBitfile" Type="Str">C:\Users\abs21004\Downloads\FPGA Bitfiles\exo1_FPGATarget_4xFFT_9F7UidU220U.lvbitx</Property>
				</Item>
				<Item Name="40 MHz Onboard Clock" Type="FPGA Base Clock">
					<Property Name="FPGA.PersistentID" Type="Str">{06FD7152-9008-47B3-8F28-696D3C97C875}</Property>
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
				<Item Name="AI_BLACKBOX" Type="VI" URL="../AI_BLACKBOX">
					<Property Name="configString.guid" Type="Str">{036DF3CB-3458-47DA-90D2-ACEBD7625A99}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{04A71BF2-AA06-4E1D-A1F1-4CCE89F7B91E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{06FD7152-9008-47B3-8F28-696D3C97C875}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{0A28A788-8D2F-4F77-B8F4-B02412C9E51D}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{0DFF9EEA-2F3A-40A7-B72E-7160789DC629}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{109438B6-C5C8-460D-8842-0C730A9DB172}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{16180A68-244E-4ABC-882E-1FE9B2B5A96C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{296720F7-2657-4D43-A0A9-060C6F1F71B7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{2A5A8526-2503-4447-A9D2-D0F9BB7F931B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{2CBA67DC-957B-4396-8C20-436E7DD8E037}resource=/AccelerationX;0;ReadMethodType=I16{2EA5272D-1487-41D3-BC4B-1436CDBE0846}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{30579913-32F8-4529-86E4-2BB0188209FE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{30C62109-A65A-415B-A8AA-34215F05DCA1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{30D2C326-733D-4A43-95E8-11C81988D3FC}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{3370460E-8621-42F8-90B0-212D3317FD61}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{362EC810-C30D-44C2-A76B-45D223D88E60}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{37044D48-48FC-4F11-808B-EF94C581D0F5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{378FE4D5-976C-49EC-839F-6CBA8EFEC038}resource=/AnalogIn1;0;ReadMethodType=U16{38760416-0184-4B83-B111-585BFAA5A42C}resource=/AnalogIn2;0;ReadMethodType=U16{3AF87085-C2D1-41BC-A2A9-56CDB792EC80}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3B381BAA-6D28-49B6-8DC5-15FDD5C4BF49}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{3C8D7016-5E0D-4EE7-A1AC-287832A90691}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{3DC9B286-C7F9-45E1-8A13-2E2ABAB62F02}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{3EFB491E-CC90-4707-97F1-CEF3377DE51E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{438FBFE9-1175-4302-8799-ABE13FADC1E9}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{459CD608-BCEF-4C73-A04E-2F6C62DA6B70}resource=/MXP/AI3;0;ReadMethodType=U16{46600E50-6CB1-473D-B999-CB5E5F92C1B3}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{4912ABC2-A372-4177-BA59-CF0D7E0A66C5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{51D565AA-225D-4DE8-B2BB-BB4D09B6344F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{525B2E61-0035-4646-A684-F06840DAD30F}resource=/MXP/AO1;0;WriteMethodType=U16{5E523F57-FC5A-4CB0-B6B7-53410322A70B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{6299C9A5-9E6C-4202-8AA1-6C9234EBC8CC}resource=/MXP/AI0;0;ReadMethodType=U16{6311591E-2529-4E29-A5EC-E25215AA2800}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{647515CF-4FD6-44AE-940A-593D8B0F3F54}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{65FC489C-2D76-48D1-B84D-8088BAA3E1FE}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{662271A6-E297-43C1-B67B-8CBEC0308530}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{66D48B2E-EF8B-487B-986E-B9AB947B0A2C}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{6C85F277-497E-42F5-A56A-4D23C4FABC76}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{7285CC5C-BF99-4FA4-9684-DFA7CE8FD910}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{72DCE64E-9177-492B-BD12-80E1EEBEF5A0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{73D8FDCB-00D6-4147-AD76-5D8D0ECC6F64}resource=/MXP/AI1;0;ReadMethodType=U16{778C9CFE-6CCB-4AF1-AF7C-AAEB85FDEB84}resource=/AnalogIn3;0;ReadMethodType=U16{7F8F077D-C47A-4410-BE5E-B5CE9B625C96}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{83F1A921-DD61-43B5-B66A-0E1E0C19D92E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{83FBA6E6-1D0E-42E1-A48E-ECCD892EA9DF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{86E46AFA-83C7-4773-94F6-DDF1CFDA8D13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{87BFE617-D994-4FB1-AD0B-06D368560E1B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{8AA92FF1-2E57-4E8F-A1AF-E76166C7A929}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{8C0967AE-99BE-4CCB-8E41-CF5B82FA0448}resource=/AccelerationY;0;ReadMethodType=I16{8D67E739-75D7-408C-9DA2-344640F6B590}resource=/MXP/AO0;0;WriteMethodType=U16{8FAE59E8-FC01-4BD5-93D4-B2129CEB25A9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{905E4E04-9B97-4455-826C-358C303A66D9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{94BC5F9E-84CE-4335-B757-979F4963B020}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{9728004D-D9C4-4B4F-A67E-CC26BA4EBCEF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{97CAF5F2-DC18-4632-9E68-808A91D820FC}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{9A784D1F-1C3F-4189-B170-9FD46FFCAD4F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{9C3CDA6D-737A-4224-9FD8-A0ADE35D2391}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{9C67454B-2364-4CDD-B7E0-16E4CD604A12}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{9E8DD9B6-C50C-49E2-A4E4-9D083DE1D83D}resource=/AnalogIn0;0;ReadMethodType=U16{A3D0E05D-9F7B-465A-B624-11E7DADD99C5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{A84CE67F-F21D-4207-992B-891D9CFA7E4A}resource=/AccelerationZ;0;ReadMethodType=I16{AC8D2DAE-59CB-4752-8421-E8EABF034750}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B3CCC653-1E16-4D31-A352-43B3B99AB2B5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{B74D8E39-1031-4249-A233-2313A21CF8B6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB9A8698-091A-487B-894F-C8F9A096902C}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{BBE1574D-BCFB-45D7-B9C6-034ABA8CDA19}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{C1FD649E-ED1B-4636-9FA1-81E2424A0B5B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{C65D4E96-3027-4822-BBE4-8672ECE90161}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{C7F5AB61-2A34-470A-B642-31F2C9D4D4CA}resource=/MXP/AI2;0;ReadMethodType=U16{DEA72476-F7DD-47DB-B9F6-C38A618AC0B9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{DF7B2D72-6610-43C2-B410-502A228D3D55}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{DF9C85A7-54C3-4A96-89E4-6963493CC050}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{E3DB5321-3865-4D01-BB41-8E05EEEE615B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{E7C586F7-C290-4CD8-993F-4B8487E6A2BB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{F0D73D2A-511D-4CE1-8A15-83E49BA6EF94}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{F57A09EE-EA36-4D3E-87AD-CC4FF3BED1A5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{F677DA21-0D2D-446C-8002-2905D115F9F2}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{FA805930-C536-4C9C-8E24-5951CF15D0D4}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
					<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
				</Item>
				<Item Name="DataIn.vi" Type="VI" URL="../DataIn.vi">
					<Property Name="configString.guid" Type="Str">{036DF3CB-3458-47DA-90D2-ACEBD7625A99}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{04A71BF2-AA06-4E1D-A1F1-4CCE89F7B91E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{06FD7152-9008-47B3-8F28-696D3C97C875}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{0A28A788-8D2F-4F77-B8F4-B02412C9E51D}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{0DFF9EEA-2F3A-40A7-B72E-7160789DC629}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{109438B6-C5C8-460D-8842-0C730A9DB172}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{16180A68-244E-4ABC-882E-1FE9B2B5A96C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{296720F7-2657-4D43-A0A9-060C6F1F71B7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{2A5A8526-2503-4447-A9D2-D0F9BB7F931B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{2CBA67DC-957B-4396-8C20-436E7DD8E037}resource=/AccelerationX;0;ReadMethodType=I16{2EA5272D-1487-41D3-BC4B-1436CDBE0846}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{30579913-32F8-4529-86E4-2BB0188209FE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{30C62109-A65A-415B-A8AA-34215F05DCA1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{30D2C326-733D-4A43-95E8-11C81988D3FC}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{3370460E-8621-42F8-90B0-212D3317FD61}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{362EC810-C30D-44C2-A76B-45D223D88E60}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{37044D48-48FC-4F11-808B-EF94C581D0F5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{378FE4D5-976C-49EC-839F-6CBA8EFEC038}resource=/AnalogIn1;0;ReadMethodType=U16{38760416-0184-4B83-B111-585BFAA5A42C}resource=/AnalogIn2;0;ReadMethodType=U16{3AF87085-C2D1-41BC-A2A9-56CDB792EC80}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3B381BAA-6D28-49B6-8DC5-15FDD5C4BF49}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{3C8D7016-5E0D-4EE7-A1AC-287832A90691}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{3DC9B286-C7F9-45E1-8A13-2E2ABAB62F02}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{3EFB491E-CC90-4707-97F1-CEF3377DE51E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{438FBFE9-1175-4302-8799-ABE13FADC1E9}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{459CD608-BCEF-4C73-A04E-2F6C62DA6B70}resource=/MXP/AI3;0;ReadMethodType=U16{46600E50-6CB1-473D-B999-CB5E5F92C1B3}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{4912ABC2-A372-4177-BA59-CF0D7E0A66C5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{51D565AA-225D-4DE8-B2BB-BB4D09B6344F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{525B2E61-0035-4646-A684-F06840DAD30F}resource=/MXP/AO1;0;WriteMethodType=U16{5E523F57-FC5A-4CB0-B6B7-53410322A70B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{6299C9A5-9E6C-4202-8AA1-6C9234EBC8CC}resource=/MXP/AI0;0;ReadMethodType=U16{6311591E-2529-4E29-A5EC-E25215AA2800}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{647515CF-4FD6-44AE-940A-593D8B0F3F54}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{65FC489C-2D76-48D1-B84D-8088BAA3E1FE}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{662271A6-E297-43C1-B67B-8CBEC0308530}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{66D48B2E-EF8B-487B-986E-B9AB947B0A2C}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{6C85F277-497E-42F5-A56A-4D23C4FABC76}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{7285CC5C-BF99-4FA4-9684-DFA7CE8FD910}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{72DCE64E-9177-492B-BD12-80E1EEBEF5A0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{73D8FDCB-00D6-4147-AD76-5D8D0ECC6F64}resource=/MXP/AI1;0;ReadMethodType=U16{778C9CFE-6CCB-4AF1-AF7C-AAEB85FDEB84}resource=/AnalogIn3;0;ReadMethodType=U16{7F8F077D-C47A-4410-BE5E-B5CE9B625C96}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{83F1A921-DD61-43B5-B66A-0E1E0C19D92E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{83FBA6E6-1D0E-42E1-A48E-ECCD892EA9DF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{86E46AFA-83C7-4773-94F6-DDF1CFDA8D13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{87BFE617-D994-4FB1-AD0B-06D368560E1B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{8AA92FF1-2E57-4E8F-A1AF-E76166C7A929}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{8C0967AE-99BE-4CCB-8E41-CF5B82FA0448}resource=/AccelerationY;0;ReadMethodType=I16{8D67E739-75D7-408C-9DA2-344640F6B590}resource=/MXP/AO0;0;WriteMethodType=U16{8FAE59E8-FC01-4BD5-93D4-B2129CEB25A9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{905E4E04-9B97-4455-826C-358C303A66D9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{94BC5F9E-84CE-4335-B757-979F4963B020}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{9728004D-D9C4-4B4F-A67E-CC26BA4EBCEF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{97CAF5F2-DC18-4632-9E68-808A91D820FC}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{9A784D1F-1C3F-4189-B170-9FD46FFCAD4F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{9C3CDA6D-737A-4224-9FD8-A0ADE35D2391}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{9C67454B-2364-4CDD-B7E0-16E4CD604A12}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{9E8DD9B6-C50C-49E2-A4E4-9D083DE1D83D}resource=/AnalogIn0;0;ReadMethodType=U16{A3D0E05D-9F7B-465A-B624-11E7DADD99C5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{A84CE67F-F21D-4207-992B-891D9CFA7E4A}resource=/AccelerationZ;0;ReadMethodType=I16{AC8D2DAE-59CB-4752-8421-E8EABF034750}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B3CCC653-1E16-4D31-A352-43B3B99AB2B5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{B74D8E39-1031-4249-A233-2313A21CF8B6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB9A8698-091A-487B-894F-C8F9A096902C}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{BBE1574D-BCFB-45D7-B9C6-034ABA8CDA19}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{C1FD649E-ED1B-4636-9FA1-81E2424A0B5B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{C65D4E96-3027-4822-BBE4-8672ECE90161}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{C7F5AB61-2A34-470A-B642-31F2C9D4D4CA}resource=/MXP/AI2;0;ReadMethodType=U16{DEA72476-F7DD-47DB-B9F6-C38A618AC0B9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{DF7B2D72-6610-43C2-B410-502A228D3D55}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{DF9C85A7-54C3-4A96-89E4-6963493CC050}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{E3DB5321-3865-4D01-BB41-8E05EEEE615B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{E7C586F7-C290-4CD8-993F-4B8487E6A2BB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{F0D73D2A-511D-4CE1-8A15-83E49BA6EF94}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{F57A09EE-EA36-4D3E-87AD-CC4FF3BED1A5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{F677DA21-0D2D-446C-8002-2905D115F9F2}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{FA805930-C536-4C9C-8E24-5951CF15D0D4}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
					<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
				</Item>
				<Item Name="EXO1.vi" Type="VI" URL="../EXO1.vi">
					<Property Name="configString.guid" Type="Str">{036DF3CB-3458-47DA-90D2-ACEBD7625A99}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=bool{04A71BF2-AA06-4E1D-A1F1-4CCE89F7B91E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=bool{06FD7152-9008-47B3-8F28-696D3C97C875}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{0A28A788-8D2F-4F77-B8F4-B02412C9E51D}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{0DFF9EEA-2F3A-40A7-B72E-7160789DC629}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=bool{109438B6-C5C8-460D-8842-0C730A9DB172}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{16180A68-244E-4ABC-882E-1FE9B2B5A96C}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=bool{296720F7-2657-4D43-A0A9-060C6F1F71B7}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=bool{2A5A8526-2503-4447-A9D2-D0F9BB7F931B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=bool{2CBA67DC-957B-4396-8C20-436E7DD8E037}resource=/AccelerationX;0;ReadMethodType=I16{2EA5272D-1487-41D3-BC4B-1436CDBE0846}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{30579913-32F8-4529-86E4-2BB0188209FE}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=bool{30C62109-A65A-415B-A8AA-34215F05DCA1}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{30D2C326-733D-4A43-95E8-11C81988D3FC}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=bool{3370460E-8621-42F8-90B0-212D3317FD61}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=bool{362EC810-C30D-44C2-A76B-45D223D88E60}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=bool{37044D48-48FC-4F11-808B-EF94C581D0F5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=bool{378FE4D5-976C-49EC-839F-6CBA8EFEC038}resource=/AnalogIn1;0;ReadMethodType=U16{38760416-0184-4B83-B111-585BFAA5A42C}resource=/AnalogIn2;0;ReadMethodType=U16{3AF87085-C2D1-41BC-A2A9-56CDB792EC80}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=bool{3B381BAA-6D28-49B6-8DC5-15FDD5C4BF49}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{3C8D7016-5E0D-4EE7-A1AC-287832A90691}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=bool{3DC9B286-C7F9-45E1-8A13-2E2ABAB62F02}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=bool{3EFB491E-CC90-4707-97F1-CEF3377DE51E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=bool{438FBFE9-1175-4302-8799-ABE13FADC1E9}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=bool{459CD608-BCEF-4C73-A04E-2F6C62DA6B70}resource=/MXP/AI3;0;ReadMethodType=U16{46600E50-6CB1-473D-B999-CB5E5F92C1B3}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8{4912ABC2-A372-4177-BA59-CF0D7E0A66C5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{51D565AA-225D-4DE8-B2BB-BB4D09B6344F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=bool{525B2E61-0035-4646-A684-F06840DAD30F}resource=/MXP/AO1;0;WriteMethodType=U16{5E523F57-FC5A-4CB0-B6B7-53410322A70B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=bool{6299C9A5-9E6C-4202-8AA1-6C9234EBC8CC}resource=/MXP/AI0;0;ReadMethodType=U16{6311591E-2529-4E29-A5EC-E25215AA2800}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=bool{647515CF-4FD6-44AE-940A-593D8B0F3F54}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=bool{65FC489C-2D76-48D1-B84D-8088BAA3E1FE}NumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool{662271A6-E297-43C1-B67B-8CBEC0308530}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=bool{66D48B2E-EF8B-487B-986E-B9AB947B0A2C}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=bool{6C85F277-497E-42F5-A56A-4D23C4FABC76}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=bool{7285CC5C-BF99-4FA4-9684-DFA7CE8FD910}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=bool{72DCE64E-9177-492B-BD12-80E1EEBEF5A0}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=bool{73D8FDCB-00D6-4147-AD76-5D8D0ECC6F64}resource=/MXP/AI1;0;ReadMethodType=U16{778C9CFE-6CCB-4AF1-AF7C-AAEB85FDEB84}resource=/AnalogIn3;0;ReadMethodType=U16{7F8F077D-C47A-4410-BE5E-B5CE9B625C96}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=bool{83F1A921-DD61-43B5-B66A-0E1E0C19D92E}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=bool{83FBA6E6-1D0E-42E1-A48E-ECCD892EA9DF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=bool{86E46AFA-83C7-4773-94F6-DDF1CFDA8D13}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=bool{87BFE617-D994-4FB1-AD0B-06D368560E1B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=bool{8AA92FF1-2E57-4E8F-A1AF-E76166C7A929}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=bool{8C0967AE-99BE-4CCB-8E41-CF5B82FA0448}resource=/AccelerationY;0;ReadMethodType=I16{8D67E739-75D7-408C-9DA2-344640F6B590}resource=/MXP/AO0;0;WriteMethodType=U16{8FAE59E8-FC01-4BD5-93D4-B2129CEB25A9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=bool{905E4E04-9B97-4455-826C-358C303A66D9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{94BC5F9E-84CE-4335-B757-979F4963B020}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=bool{9728004D-D9C4-4B4F-A67E-CC26BA4EBCEF}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=bool{97CAF5F2-DC18-4632-9E68-808A91D820FC}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{9A784D1F-1C3F-4189-B170-9FD46FFCAD4F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=bool{9C3CDA6D-737A-4224-9FD8-A0ADE35D2391}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=bool{9C67454B-2364-4CDD-B7E0-16E4CD604A12}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{9E8DD9B6-C50C-49E2-A4E4-9D083DE1D83D}resource=/AnalogIn0;0;ReadMethodType=U16{A3D0E05D-9F7B-465A-B624-11E7DADD99C5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=bool{A84CE67F-F21D-4207-992B-891D9CFA7E4A}resource=/AccelerationZ;0;ReadMethodType=I16{AC8D2DAE-59CB-4752-8421-E8EABF034750}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=bool{B3CCC653-1E16-4D31-A352-43B3B99AB2B5}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=bool{B74D8E39-1031-4249-A233-2313A21CF8B6}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{BB9A8698-091A-487B-894F-C8F9A096902C}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=bool{BBE1574D-BCFB-45D7-B9C6-034ABA8CDA19}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=bool{C1FD649E-ED1B-4636-9FA1-81E2424A0B5B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=bool{C65D4E96-3027-4822-BBE4-8672ECE90161}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=bool{C7F5AB61-2A34-470A-B642-31F2C9D4D4CA}resource=/MXP/AI2;0;ReadMethodType=U16{DEA72476-F7DD-47DB-B9F6-C38A618AC0B9}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8{DF7B2D72-6610-43C2-B410-502A228D3D55}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=bool{DF9C85A7-54C3-4A96-89E4-6963493CC050}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=bool{E3DB5321-3865-4D01-BB41-8E05EEEE615B}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=bool{E7C586F7-C290-4CD8-993F-4B8487E6A2BB}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=bool{F0D73D2A-511D-4CE1-8A15-83E49BA6EF94}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=bool{F57A09EE-EA36-4D3E-87AD-CC4FF3BED1A5}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=bool{F677DA21-0D2D-446C-8002-2905D115F9F2}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=bool{FA805930-C536-4C9C-8E24-5951CF15D0D4}ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGA</Property>
					<Property Name="configString.name" Type="Str">40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000,000000;MaxFreq=40000000,000000;VariableFreq=0;NomFreq=40000000,000000;PeakPeriodJitter=250,000000;MinDutyCycle=50,000000;MaxDutyCycle=50,000000;Accuracy=100,000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;AccelerationXresource=/AccelerationX;0;ReadMethodType=I16AccelerationYresource=/AccelerationY;0;ReadMethodType=I16AccelerationZresource=/AccelerationZ;0;ReadMethodType=I16AnalogIn0resource=/AnalogIn0;0;ReadMethodType=U16AnalogIn1resource=/AnalogIn1;0;ReadMethodType=U16AnalogIn2resource=/AnalogIn2;0;ReadMethodType=U16AnalogIn3resource=/AnalogIn3;0;ReadMethodType=U16Comm LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Green;0;ReadMethodType=bool;WriteMethodType=boolComm LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Comm LED Red;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO0;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO1;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO2;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO3;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO4;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO5;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO6;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO7;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO8;0;ReadMethodType=bool;WriteMethodType=boolDIO/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/DIO/DIO9;0;ReadMethodType=bool;WriteMethodType=boolI2C SCLArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SCL;0;ReadMethodType=bool;WriteMethodType=boolI2C SDAArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/I2C SDA;0;ReadMethodType=bool;WriteMethodType=boolMode LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Green;0;ReadMethodType=bool;WriteMethodType=boolMode LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Mode LED Red;0;ReadMethodType=bool;WriteMethodType=boolMXP/AI0resource=/MXP/AI0;0;ReadMethodType=U16MXP/AI1resource=/MXP/AI1;0;ReadMethodType=U16MXP/AI2resource=/MXP/AI2;0;ReadMethodType=U16MXP/AI3resource=/MXP/AI3;0;ReadMethodType=U16MXP/AO0resource=/MXP/AO0;0;WriteMethodType=U16MXP/AO1resource=/MXP/AO1;0;WriteMethodType=U16MXP/DIO0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO0;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO10ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO10;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO11ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO11;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO12ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO12;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO13ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO13;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO14ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO14;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO15:8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15:8;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO15ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO15;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO1;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO2;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO3;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO4ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO4;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO5ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO5;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO6ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO6;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO7:0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7:0;0;ReadMethodType=u8;WriteMethodType=u8MXP/DIO7ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO7;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO8ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO8;0;ReadMethodType=bool;WriteMethodType=boolMXP/DIO9ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/MXP/DIO9;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM0;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM1;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM2;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM3;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM4ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM4;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM5ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM5;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM6ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM6;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM7ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM7;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM8ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM8;0;ReadMethodType=bool;WriteMethodType=boolPWM/PWM9ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/PWM/PWM9;0;ReadMethodType=bool;WriteMethodType=boolRadio LED GreenArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Green;0;ReadMethodType=bool;WriteMethodType=boolRadio LED RedArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/Radio LED Red;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd0;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd1;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd2;0;ReadMethodType=bool;WriteMethodType=boolRelayFwd3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayFwd3;0;ReadMethodType=bool;WriteMethodType=boolRelayRev0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev0;0;ReadMethodType=bool;WriteMethodType=boolRelayRev1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev1;0;ReadMethodType=bool;WriteMethodType=boolRelayRev2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev2;0;ReadMethodType=bool;WriteMethodType=boolRelayRev3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RelayRev3;0;ReadMethodType=bool;WriteMethodType=boolroboRIO/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSROBORIOFPGA_TARGET_FAMILYZYNQTARGET_TYPEFPGARSLArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/RSL;0;ReadMethodType=bool;WriteMethodType=boolSPI CLKArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CLK;0;ReadMethodType=bool;WriteMethodType=boolSPI CS0ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS0;0;ReadMethodType=bool;WriteMethodType=boolSPI CS1ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS1;0;ReadMethodType=bool;WriteMethodType=boolSPI CS2ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS2;0;ReadMethodType=bool;WriteMethodType=boolSPI CS3ArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI CS3;0;ReadMethodType=bool;WriteMethodType=boolSPI MISOArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MISO;0;ReadMethodType=bool;WriteMethodType=boolSPI MOSIArbitrationForOutputData=NeverArbitrate;ArbitrationForOutputEnable=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForOutputEnable=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/SPI MOSI;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolUser ButtonNumberOfSyncRegistersForReadInProject=Auto;resource=/User Button;0;ReadMethodType=bool</Property>
					<Property Name="NI.LV.FPGA.InterfaceBitfile" Type="Str">C:\Users\abs21004\Downloads\FPGA Bitfiles\exo1_FPGATarget_EXO1_-qY0LurneMw.lvbitx</Property>
				</Item>
				<Item Name="Dependencies" Type="Dependencies">
					<Item Name="vi.lib" Type="Folder">
						<Item Name="LVFixedPointQuantizationPolicyTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/fxp/LVFixedPointQuantizationPolicyTypeDef.ctl"/>
					</Item>
				</Item>
				<Item Name="Build Specifications" Type="Build">
					<Item Name="4xFFT" Type="{F4C5E96F-7410-48A5-BB87-3559BC9B167F}">
						<Property Name="AllowEnableRemoval" Type="Bool">false</Property>
						<Property Name="BuildSpecDecription" Type="Str"></Property>
						<Property Name="BuildSpecName" Type="Str">4xFFT</Property>
						<Property Name="Comp.BitfileName" Type="Str">exo1_FPGATarget_4xFFT_9F7UidU220U.lvbitx</Property>
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
						<Property Name="ProjectPath" Type="Path">/C/Users/abs21004/Downloads/EXO1.lvproj</Property>
						<Property Name="RelativePath" Type="Bool">true</Property>
						<Property Name="RunWhenLoaded" Type="Bool">false</Property>
						<Property Name="SupportDownload" Type="Bool">true</Property>
						<Property Name="SupportResourceEstimation" Type="Bool">false</Property>
						<Property Name="TargetName" Type="Str">FPGA Target</Property>
						<Property Name="TopLevelVI" Type="Ref">/My Computer/RoboRIO/FPGA Target/4xFFT.vi</Property>
					</Item>
					<Item Name="DataIn" Type="{F4C5E96F-7410-48A5-BB87-3559BC9B167F}">
						<Property Name="AllowEnableRemoval" Type="Bool">false</Property>
						<Property Name="BuildSpecDecription" Type="Str"></Property>
						<Property Name="BuildSpecName" Type="Str">DataIn</Property>
						<Property Name="Comp.BitfileName" Type="Str">exo1_FPGATarget_DataIn_4vMXAs4II6k.lvbitx</Property>
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
						<Property Name="ProjectPath" Type="Path">/C/Users/abs21004/Downloads/EXO1.lvproj</Property>
						<Property Name="RelativePath" Type="Bool">true</Property>
						<Property Name="RunWhenLoaded" Type="Bool">false</Property>
						<Property Name="SupportDownload" Type="Bool">true</Property>
						<Property Name="SupportResourceEstimation" Type="Bool">false</Property>
						<Property Name="TargetName" Type="Str">FPGA Target</Property>
						<Property Name="TopLevelVI" Type="Ref">/My Computer/RoboRIO/FPGA Target/DataIn.vi</Property>
					</Item>
					<Item Name="EXO1" Type="{F4C5E96F-7410-48A5-BB87-3559BC9B167F}">
						<Property Name="AllowEnableRemoval" Type="Bool">false</Property>
						<Property Name="BuildSpecDecription" Type="Str"></Property>
						<Property Name="BuildSpecName" Type="Str">EXO1</Property>
						<Property Name="Comp.BitfileName" Type="Str">exo1_FPGATarget_EXO1_-qY0LurneMw.lvbitx</Property>
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
						<Property Name="NI.LV.FPGA.LastCompiledBitfilePath" Type="Path">/C/Users/abs21004/Downloads/FPGA Bitfiles/exo1_FPGATarget_EXO1_-qY0LurneMw.lvbitx</Property>
						<Property Name="NI.LV.FPGA.LastCompiledBitfilePathRelativeToProject" Type="Path">FPGA Bitfiles/exo1_FPGATarget_EXO1_-qY0LurneMw.lvbitx</Property>
						<Property Name="ProjectPath" Type="Path">/C/Users/abs21004/Downloads/EXO1.lvproj</Property>
						<Property Name="RelativePath" Type="Bool">true</Property>
						<Property Name="RunWhenLoaded" Type="Bool">false</Property>
						<Property Name="SupportDownload" Type="Bool">true</Property>
						<Property Name="SupportResourceEstimation" Type="Bool">false</Property>
						<Property Name="TargetName" Type="Str">FPGA Target</Property>
						<Property Name="TopLevelVI" Type="Ref">/My Computer/RoboRIO/FPGA Target/EXO1.vi</Property>
					</Item>
				</Item>
			</Item>
		</Item>
		<Item Name="Dependencies" Type="Dependencies"/>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
