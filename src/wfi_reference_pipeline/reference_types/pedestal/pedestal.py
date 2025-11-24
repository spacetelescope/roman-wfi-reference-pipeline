from ..reference_type import ReferenceType

from wfi_reference_pipeline.resources.wfi_meta_pedestal import WFIMetaPedestal

class Pedestal(ReferenceType):

    def __init__(
        self,
        meta_data,
        file_list=None,
        ref_type_data=None,
        bit_mask=None,
        outfile="roman_pedestal.asdf",
        clobber=False,
    ):
        
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber
        )

        if not isinstance(meta_data, WFIMetaPedestal):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaPedestal"
            )
        
    def calculate_error(self):
        pass # Adding abstract methods, not applicable for now

    def update_data_quality_array(self):
        pass # Adding abstract methods, not applicable for now

    def populate_datamodel_tree(self):
        pass # Adding abstract methods, not applicable for now