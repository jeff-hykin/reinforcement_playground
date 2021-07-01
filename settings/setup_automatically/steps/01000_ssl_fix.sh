# if file exists, use it
export SSL_CERT_FILE="$PROJECTR_FOLDER/resources/cacert.pem"
# for some reason git needs its own var 
export GIT_SSL_CAINFO="$SSL_CERT_FILE"