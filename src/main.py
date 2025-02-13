from GNNTrainer import GNNTrainer
from DataConversion import RDFGraphConverter

# Usage example
converter = RDFGraphConverter("AIFB")
hetero_data = converter.load_dataset()
print(converter.get_statistics())

print(hetero_data)

gnn = GNNTrainer(hetero_data,'Person')
gnn.train()
# gnn.save_model()
# print(gnn.get_positive_nodes())
# print(gnn.get_negative_nodes())

# gnn.load_model()
gnn.evaluate_test_set()