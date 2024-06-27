# EXO1
This is an example change
This is a new example change
This is a second example change
This is now the third example change
This is now a fourth example change
This is a fifth example change


# Labview
This Labview project is sectioned off into 3 parts. The Real-Time, FPGA & Preprocessing parts. Each part files and objective is outline below

The project and the deploying of the RoboRIO is handled by the .Ivproj file and is pure black magic. As of writing it only works on a singular machine with the help of Daniel Morberg. This does not apply to subVi's which any machine can made. NOTE: use only modules the RoboRIO is compatible with. 

Just know that any files underneath the "roboRIO-330-FRC" is deployed on the Real-Time part and everything underneath the "FPGA Target (RIO0,roboRIO)" is deployed on the FPGA side.

## Real-Time
Any files with the prefix RT-* is part of and deployed toward the Realtime section of the RoboRIO. 

- RT-target - single rate.vi: This is the main file. It contains the overall loop structure along with the initialisation step and a termination step if the stop button is pressed.

- RT-Init.vi: This is the initialisation file. The FPGA file, Weights and settings for the motorcontroller are all set up here

- RT-Data-in.vi: This is the main data collection file. It facilitates a sliding window of variable size. 

- RT-RNN.vi: This is the main Recurrent neural network file. It breaks up the weights cluster and inputs them into layers

- RT-Mode-Selection.vi: This is the selection module which picks the highest of the catagories and outputs a u64 accordingly

- RT-Terminate.vi: This closes and error handles the CAN bus and the FPGA module

- RT-Motor-Controls.vi: This takes a u64 input, splits it into 8 u8 to fit the bitstream and pushes it to a CAN-Write module then reads for an ACK.

- RT-RIO-to-Volt.vi: This a help function to translate (Assumed to be linearly) from the Roborio FPGA output to Volts. Calibrated by 4000:5Volts

- RT-LSTM-Layer.vi: This function is a loop to iterate through the LSTM layer and saving each nodes output states and feeds them back in the next iteration

- RT-ANN-Node: This function loops through the input layer where the last collumn is the bias

- RT-Read-from-Bin: This is a help function inside "RT-init.vi" to read the weight matrices from a Binary File

- RT-LSTM.vi: This is the core part of the RNN with each node containing a "input gate", "output gate" & a "forget gate"

- RT-sigmoid.vi: A simple help function to faciliate a sigmoid curve outside of an expression node

- RT-Tanh.vi: A simple help function to faciliate a tanh curve outside of an expression node 

## FPGA
Any files with the prefix FPGA-* is part of and deployed toward the FPGA section of the RoboRIO. 

- FPGA-DataIn: A simple setup to start reading from the AnalogIn nodes on the FPGA

## Preproccessing
Any files lacking a prefix is to be ran on a regular machine instead of the embedded system

- 12Weights-to-4Weights.vi: A half implemented attempt to translate our XML created weights to the weights used elsewhere in the system

- Write-To-Bin.vi: This takes in the four weight matrices and turns them into Binary files to be read later

- XML-to-Array.vi: Parses our XML file into Arrays
