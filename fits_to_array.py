import sys



from astropy.io import fits
import struct

input_ = sys.argv[1]
output_ = sys.argv[2]



hdulist = fits.open(input_ )

scidata = hdulist[1].data
out_file = open(output_,"wb")
data = [scidata.shape[0], scidata.shape[1]]
out_file.write(struct.pack('i'*len(data), *data))
tmp = []
for i in range(0, ( scidata.shape[0])):
	for j in range(0, ( scidata.shape[1])):
		tmp.append(scidata[i, j])
out_file.write(struct.pack('d'*len(tmp), *tmp))



out_file.close()



