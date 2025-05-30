from .SRC import l0decoder

def meta_extractor(dat_filepath):
    """
    Function auxiliary to extract metadata with richall processing.
    Temporary func. 
    """
    
    metadecoder = l0decoder.Level0Decoder(filename=dat_filepath)
    return metadecoder.decode_metadata()