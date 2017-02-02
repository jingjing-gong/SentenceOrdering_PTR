# SentenceOrdering_PTR
Sentence ordering using pointer network
### how to get it to work
after data retrieved in all_data directory, and embedding files put in all_data/embed directory as specified. run: 

`python model.py --load-config --weight-path ./final_save/lstm`

`python model.py --load-config --weight-path ./final_save/cbow`

`python model.py --load-config --weight-path ./final_save/cnn`

>threre are default config files in `./final_save/<model>` dir.
>you can run: 

>`python model.py --weight-path ./your/save/path` 

>to generate config file and modify it as you want. then run:

>`python model.py --load-config --weight-path ./your/save/path` 

>to run the model you configured through modifying *config* file

