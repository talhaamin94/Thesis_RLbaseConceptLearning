# import os
# import torch
# import pandas as pd

# class TransformLabels:
#     def __init__(self, dataset_name):
#         self.dataset_name = dataset_name.lower()  # Ensure lowercase for consistency

#     def assign_labels(self, train_file, test_file):
#         """Assigns labels based on the dataset type."""
#         if self.dataset_name == "aifb":
#             return self._assign_labels_aifb(train_file, test_file)
#         elif self.dataset_name == "mutag":
#             return self._assign_labels_mutag(train_file, test_file)
#         else:
#             raise ValueError(f"Dataset {self.dataset_name} not supported for label transformation.")

#     def _assign_labels_aifb(self, train_file, test_file):
#         """Assigns labels for AIFB dataset based on the largest research group, with debugging outputs."""
#         train_df = pd.read_csv(train_file, sep="\t")
#         test_df = pd.read_csv(test_file, sep="\t")

#         # Count the number of research groups and their member counts
#         research_group_counts = train_df["label_affiliation"].value_counts()

#         # print("\n[DEBUG] Research Group Statistics:")
#         # print(f"Total Research Groups: {len(research_group_counts)}")
#         # print("Members per Research Group:")
#         # print(research_group_counts)

#         # Identify the largest research group
#         largest_group = research_group_counts.idxmax()
#         # print(f"\n[DEBUG] Largest Research Group: {largest_group} with {research_group_counts.max()} members")

#         # Assign new labels: 1 for the largest group, 0 for the rest
#         train_df["new_label"] = (train_df["label_affiliation"] == largest_group).astype(int)
#         test_df["new_label"] = (test_df["label_affiliation"] == largest_group).astype(int)

#         return train_df, test_df


#     def _assign_labels_mutag(self, train_file, test_file):
#         """Assigns labels for MUTAG dataset based on mutagenicity classification."""
#         train_df = pd.read_csv(train_file, sep="\t")
#         test_df = pd.read_csv(test_file, sep="\t")

#         # Assume 'mutagenicity' is the column indicating mutagenic compounds (1) and non-mutagenic (0)
#         if "mutagenicity" in train_df.columns:
#             train_df["new_label"] = train_df["mutagenicity"]
#         else:
#             raise KeyError("Column 'mutagenicity' not found in MUTAG dataset.")

#         if "mutagenicity" in test_df.columns:
#             test_df["new_label"] = test_df["mutagenicity"]
#         else:
#             raise KeyError("Column 'mutagenicity' not found in MUTAG dataset.")

#         return train_df, test_df

#     def transform_and_save_labels(self, train_file, test_file, train_output, test_output):
#         """Transforms the dataset labels and saves the new files in dataset-specific directories."""
#         train_df, test_df = self.assign_labels(train_file, test_file)

#         # Ensure the directory exists
#         os.makedirs(os.path.dirname(train_output), exist_ok=True)

#         # Save the updated datasets
#         train_df.to_csv(train_output, sep=",", index=False)
#         test_df.to_csv(test_output, sep=",", index=False)

import os
import torch
import pandas as pd

class TransformLabels:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name.lower()  # Ensure lowercase for consistency

    def assign_labels(self, train_file, test_file):
        """Assigns labels based on the dataset type."""
        # if self.dataset_name == "mini":
        #     return self._assign_labels_mini(train_file, test_file)
        if self.dataset_name == "aifb":
            return self._assign_labels_aifb(train_file, test_file)
        # elif self.dataset_name == "mutag":
        #     return self._assign_labels_mutag(train_file, test_file)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported for label transformation.")

    # def _assign_labels_mini(self, train_file, test_file):
    #     """Assigns labels for 'mini' dataset based on classification column."""
    #     train_df = pd.read_csv(train_file, sep="\t")
    #     test_df = pd.read_csv(test_file, sep="\t")

    #     # Convert classification labels (positive -> 1, negative -> 0)
    #     train_df["new_label"] = train_df["classification"].apply(lambda x: 1 if x == 1 else 0)
    #     test_df["new_label"] = test_df["classification"].apply(lambda x: 1 if x == 1 else 0)

    #     return train_df, test_df

    def _assign_labels_aifb(self, train_file, test_file):
        """Assigns labels for AIFB dataset based on the largest research group."""
        train_df = pd.read_csv(train_file, sep="\t")
        test_df = pd.read_csv(test_file, sep="\t")

        # Count the number of research groups and their member counts
        research_group_counts = train_df["label_affiliation"].value_counts()

        # Identify the largest research group
        largest_group = research_group_counts.idxmax()

        # Assign new labels: 1 for the largest group, 0 for the rest
        train_df["new_label"] = (train_df["label_affiliation"] == largest_group).astype(int)
        test_df["new_label"] = (test_df["label_affiliation"] == largest_group).astype(int)

        return train_df, test_df

    # def _assign_labels_mutag(self, train_file, test_file):
    #     """Assigns labels for MUTAG dataset based on mutagenicity classification."""
    #     train_df = pd.read_csv(train_file, sep="\t")
    #     test_df = pd.read_csv(test_file, sep="\t")

    #     # Assume 'mutagenicity' column contains labels
    #     if "mutagenicity" in train_df.columns:
    #         train_df["new_label"] = train_df["mutagenicity"]
    #     else:
    #         raise KeyError("Column 'mutagenicity' not found in MUTAG dataset.")

    #     if "mutagenicity" in test_df.columns:
    #         test_df["new_label"] = test_df["mutagenicity"]
    #     else:
    #         raise KeyError("Column 'mutagenicity' not found in MUTAG dataset.")

    #     return train_df, test_df

    def transform_and_save_labels(self, train_file, test_file, train_output, test_output):
        """Transforms the dataset labels and saves the new files in dataset-specific directories."""
        train_df, test_df = self.assign_labels(train_file, test_file)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(train_output), exist_ok=True)

        # Save the updated datasets
        train_df.to_csv(train_output, sep=",", index=False)
        test_df.to_csv(test_output, sep=",", index=False)

        # print(f"[DEBUG] Labels transformed and saved: {train_output}, {test_output}")
