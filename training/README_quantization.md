Notes on quantization/plan.


Writing this just as notes for myself and Daga as to how to integrate and 
run the quantization code.

Basically we need to set up a workflow that looks something like this. It 
needs to be easy to run so that Daga can run when he has retrained the 
student model.

0. Train teacher.

1. Distill a student model from model that has been trained succesfully.
This distilled model code should include the setup_step() definition/method
so that it can instantiate the weights for the stepped version of the model
(as opposed to the convolutional mode of the model)

2. Once we have got those weights for the stepped model (i.e. A, B, C, D,
etc). Then we need to load that into the pure stepped model which I have put in 
model/stepssm.py
This model only contains stepped weights, and can only be run in recurrent/stepped
mode. We then run an epoch of just pure inference (or several samples) to gather
statistics on the activation functions that are flowing through the model during normal operation.

3. Once we have run the gathering of the statistics of the activations going through the model,
and therefore the scaling activations, these need to then be read into the quantized model script,
which is currently in qssm.py.
This contains paramters/buffers which need to be assigned to the weights of the pure stepped model
mentioned in step 2. 


Extra notes:

I haven't completely tested the code for the models which I am uploading here, so we might need to create
some unit tests to verify the mathematical equivalence of the student model code and the stepssm.py code
and then perhaps another script to compare the outputs of the quantized model and the non quantized step
model
