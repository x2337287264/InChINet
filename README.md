# InChINet
The pre-trained model is in the valid_model folder. Please decompress and then place the valid_model folder in the predict folder.
The pretrain/tokenized_smile and pretrain/tokenized_inchi contain tokenized smile and tokenized inchi for pre-training. They can be obtained from PubChem. https://pubchem.ncbi.nlm.nih.gov/
SMILES can be tokenized using the instruction:
./fast applybpe ../tokenized_smile/1.txt ../smile-1.txt codes_smile
or
import fastBPE
bpe = fastBPE.fastBPE(codes_path, vocab_path)
bpe.apply(["Roasted barramundi fish", "Centrally managed over a client-server architecture"])
