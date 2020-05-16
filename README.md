Please read the pdf and view the slides to learn more about the project. Run the commands below (from the current directory) in the terminal to reproduce the results.

The best results are in Resuls/WithRandomness

Execute the following in terminal
1. `python machine_learning/embeddings/embed_fruits.py` to make the embeddings.
2. `python machine_learning/discriminator/classifier.py` to train the classifier
3. `python -m machine_learning.encoder_decoder.VAE.ConvNetVAEFruits` to train the ConvNet
4. `python -c "from machine_learning.encoder_decoder.VAE.TextEncoder import run_make_data; run_make_data()"` to make the data to pretrain the Text2Image encoding network
5. `python -m machine_learning.encoder_decoder.VAE.TextEncoder` to pretrain the Text2Image encoding network
6. `python -m machine_learning.encoder_decoder.VAE.Text2PicVAE` to train the Text2Pic VAE
6. `python -c "from machine_learning.encoder_decoder.VAE.Text2PicVAE import evaluate_results; evaluate_results()"` to evaluate the Text2PICVAE