## Notes for the FGS workflow script

The script to generate the FGS mask (both bitmask and boolean mask) is `full_workflow.py`. It takes multiple existing components of the RFP to generate intermediate products (such as a dark rate image, CDS noise image, super slope image) that are used to identify different bad pixel classes. The thresholds applied (defined at the top of the file, all in DN) are directly from GSFC. Note that the FGSMask() implmentation into the RFP will be a bit different than the other existing modules. See S. R. Gomez STScI-000863 for more details on the file format that PSS expects for this mask.

1) The mask is in detector coordinates, NOT SCIENCE! 
2) It is also a FITS file, not ASDF!
3) Finally, the mask is BOOLEAN! 0 = Good pixel, 1 = Bad pixel. The script creates both a bitmask (with the bad pixel classes and their bitvalues at the top of the script) and a boolean mask in the correct coordinate system.