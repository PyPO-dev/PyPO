from examples.DRO_PSF_PO import ex_DRO_PO
from examples.ASTE_PSF_PO import ex_ASTE_PO

def select_example():
    example = input("""Please select an example to run.\n
Possible options: DRO_PO
                  ASTE_PO

-> """)

    print(example)
    if example == "DRO_PO":
        ex_DRO_PO()

    elif example == "ASTE_PO":
        ex_ASTE_PO()

if __name__ == "__main__":
    select_example()
