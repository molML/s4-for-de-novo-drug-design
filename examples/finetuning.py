from s4dd import S4forDenovoDesign

# Load the pretrained model
s4 = S4forDenovoDesign.from_file("./demo/pretrained_model/")
# Fine-tune the model on a small subset of bioactive molecules
s4.train(
    training_molecules_path="./datasets/pkm2/train.zip",
    val_molecules_path="./datasets/pkm2/valid.zip",
)
# Save the model
s4.save("./demo/finetuned_model/")
