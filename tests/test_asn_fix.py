import das4whales as dw

filepath = r'C:\Users\ers334\Desktop\testingData\Svalbard_Josephine\2020\020549.hdf5'

metadata = dw.data_handle.get_acquisition_parameters(filepath, 'asn_alt')

print(metadata)