from s4dd import S4forDenovoDesign

# Create an S4 model
s4 = S4forDenovoDesign(
    n_max_epochs=3,  # This is for only demonstration purposes. Set this to a (much) higher value for actual training. Default: 400
    batch_size=64,  # This is for only demonstration purposes. The value in the paper is 2048.
    device="cuda",  # replace this with "cpu" if you don't have a CUDA-enabled GPU
)
# Pretrain the model on a small subset of ChEMBL
s4.train(
    training_molecules_path="./datasets/chemblv31/train.zip",
    val_molecules_path="./datasets/chemblv31/valid.zip",
)
# Save the model
s4.save("./demo/pretrained_model/")
