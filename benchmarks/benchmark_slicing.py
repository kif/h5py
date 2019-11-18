#!/usr/bin/env python3
import sys
import os
import time
import numpy
#import hdf5plugin
import h5py

# Some constants
ndim = 3
size = 1024
chunk = 64
dtype = numpy.dtype("float32")
precision = 16
nb_read = 64
workspace = "/tmp"

print(sys.executable)

total_size = size ** ndim * dtype.itemsize
needed_memory = size ** (ndim-1) * dtype.itemsize * chunk

print("Total volume size: %.3fGB, Needed memory: %.3fGB"%(total_size/1e9, needed_memory/1e9))

def create_file(directory):
    """
    :param firectory: where to create the file
    :return: the filename, the path
    """
    filename = os.path.join(directory, "benchmark_slicing.h5") 
    h5path = "data"
    shape = [size]  * ndim
    chunks = (chunk,) * ndim
    mask = numpy.uint32(((1<<32) - (1<<(24 - precision)))) 
    #options = hdf5plugin.Blosc(cname='lz4', clevel=1, shuffle=2)
    #options = hdf5plugin.Bitshuffle()
    with h5py.File(filename, "w") as h:
        ds = h.create_dataset(h5path, 
                              shape, 
                              chunks=chunks,)
                              #**options)
                              #compression = 32008,
                              #compression_opts=(0, 2))
        
        for i in range(0, size, chunk):
            x, y, z = numpy.ogrid[i:i+chunk, :size, :size]
            data = (numpy.sin(x/3)*numpy.sin(y/5)*numpy.sin(z/7)).astype(dtype)
            idata = data.view("uint32")
            idata &= mask # mask out the last 16 bits
            ds[i:i+chunk] = data
    return (filename, h5path)


def read_slice(ds, position):
    assert len(position) == ndim
    assert ds.ndim == ndim
    return ds[position[0], :, :], ds[:, position[1], :], ds[:, :, position[2]] 


def read_many_slices(filename, nb_read):
    where = numpy.random.randint(0, size, size=(nb_read, ndim))
    with h5py.File(filename, "r") as h:
        ds = h[p]
        t0 = time.time()
        for i in where:
            read_slice(ds, i)
        t1 = time.time()
    return t1 - t0


if __name__ == "__main__":
    t0 = time.time()
    fn, p = create_file(workspace)
    t1 = time.time()
    print("Filename: %s compression: %.3f "%(fn, total_size/os.stat(fn).st_size), 
          "time %.3fs"%(t1-t0), 
          "effective write speed  %.3fs MB/s"%(os.stat(fn).st_size/(t1-t0)/1e6))
    t = read_many_slices(fn, nb_read)
    print("Time for reading 3x%s slices: %.3fs fps: %.3f"%(nb_read, t1-t0, ndim*nb_read/t), 
          "Uncompressed data read speed %.3f MB/s"%(ndim*nb_read*needed_memory/t/1e6))
    
            
        
         