mkdir build
cd build

if [ $INSTALL ]; then
    cmake ..; make install
else
    cmake ..; make
fi
