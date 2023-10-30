from s4dd import S4forDenovoDesign

# Load the fine-tuned model
s4 = S4forDenovoDesign.from_file("./demo/finetuned_model/")
# Design new molecules
designs, lls = s4.design_molecules(n_designs=128, batch_size=64)

# Save the designs
with open("./demo/designs.smiles", "w") as f:
    f.write("\n".join(designs))

# Save the log-likelihoods of the designs
with open("./demo/lls.txt", "w") as f:
    f.write("\n".join([str(ll) for ll in lls]))
