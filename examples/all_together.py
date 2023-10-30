from s4dd import S4forDenovoDesign

# Create an S4 model with (almost) the same parameters as in the paper.
s4 = S4forDenovoDesign(
    n_max_epochs=3,  # This is for only demonstration purposes. Set this to a (much) higher value for actual training. Default: 400.
    batch_size=64,  # This is for only demonstration purposes. The value in the paper is 2048.
    device="cuda",  # replace this with "cpu" if you don't have a CUDA-enabled GPU.
)
# Pretrain the model on a small subset of ChEMBL
s4.train(
    training_molecules_path="./datasets/chemblv31/train.zip",
    val_molecules_path="./datasets/chemblv31/valid.zip",
)

# save the pretrained model
s4.save("./demo/pretrained_model/")

# Fine-tune the model on a small subset of bioactive molecules
s4.train(
    training_molecules_path="./datasets/pkm2/train.zip",
    val_molecules_path="./datasets/pkm2/valid.zip",
)

# save the fine-tuned model
s4.save("./demo/finetuned_model/")


# Design new molecules
designs, lls = s4.design_molecules(n_designs=128, batch_size=64, temperature=1)

# Save the designs
with open("./demo/designs.smiles", "w") as f:
    f.write("\n".join(designs))

# Save the log-likelihoods of the designs
with open("./demo/lls.txt", "w") as f:
    f.write("\n".join([str(ll) for ll in lls]))
