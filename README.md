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



## FPGA


## Preproccessing

