PB_REL="https://github.com/protocolbuffers/protobuf/releases"
curl -LO \$PB_REL/download/v3.13.0/protoc-3.13.0-linux-x86_64.zip
unzip protoc-3.13.0-linux-x86_64.zip -d \$HOME/.local
export PATH="\$PATH:\$HOME/.local/bin"
