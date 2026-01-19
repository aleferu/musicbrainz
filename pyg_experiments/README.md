# pyg_experiments

All the dataset creation and training of the models happen here.

- `train_mlp.py` trains the MLPs.
- `train.py` and `train_mb.py` are outdated, since we ended up having to use a server method due to driver issues (pyg-lib not being updated to work with CUDA 12.8 at the moment of training).
- `[test_]batch_server.py` are the data loaders that serve batches to the train scripts `batch_trainer[_lfm].py` and `batch_tester[_lfm].py`. You can use `batch_tester_results.py` to check the output of the model.
- `results_writer.py` and `results_sorter.py` handle the CSV with the results.
- `build_ds.py` build the tensors used in `build_train_heterodata[_mb].py`.
- `build_tags_has_tag_tensor_musicbrainz.py` reverts the tags to a version without LFM.
