#!/bin/bash

if [ ! -e dataSet.tgz ] 
then
    wget "https://doc-14-3c-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/0v3oorvakj8ov8hcnk9ta2h636moo4bs/1479139200000/12916301274912055673/*/0B-mnK8kniGAieXZtRmRzX2NSVDg?e=download" -O dataSet.tgz
    tar zxvf dataSet.tgz
fi

if [ ! -e vocab.tgz ] 
then
    wget "https://doc-04-1s-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/0bk0up3m3qd2ba957j6388830r8337j0/1479139200000/03531503277099621782/*/0B7_0fbjp4P-rSHhPc0hyQy1xdzA?e=download" -O vocab.tgz
    tar zxvf vocab.tgz
fi
