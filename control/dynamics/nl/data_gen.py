from control.dynamics.nl.narendra_li import create_narendra_li_datasets

dataset_train, dataset_valid, dataset_test = create_narendra_li_datasets()

dataset_train.to_csv("narendra_li_train_1.csv", index=False)
dataset_valid.to_csv("narendra_li_val_1.csv", index=False)
dataset_test.to_csv("narendra_li_test_1.csv", index=False)


