#!/usr/bin/env sh
# This script downloads the Stanford CoreNLP models.

CORENLP=stanford-corenlp-full-2015-12-09
SPICELIB=pycocoevalcap/spice/lib
JAR=stanford-corenlp-3.6.0

DIR="$( cd "$(dirname "$0")" || exit ; pwd -P )"
cd $DIR || exit

if [ -f $SPICELIB/$JAR.jar ]; then
  echo "Found Stanford CoreNLP."
else
  echo "Downloading..."
  wget http://nlp.stanford.edu/software/$CORENLP.zip
  echo "Unzipping..."
  unzip $CORENLP.zip -d $SPICELIB/
  mv $SPICELIB/$CORENLP/$JAR.jar $SPICELIB/
  mv $SPICELIB/$CORENLP/$JAR-models.jar $SPICELIB/
  rm -f $CORENLP.zip
  rm -rf $SPICELIB/$CORENLP/
  echo "Done."
fi
