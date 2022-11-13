from examples.DRO_PSF_PO import ex_DRO_PO
from examples.DRO_PSF_RT import ex_DRO_RT
from examples.ASTE_PSF_PO import ex_ASTE_PO
from examples.ASTE_PSF_RT import ex_ASTE_RT

def select_example():
    example = input("""Welcome to the POPPy example interface!
Please select an example to run.

Possible options: DRO_PO-(CPU/GPU)
                  DRO_RT-(CPU/GPU)
                  ASTE_PO-(CPU/GPU)
                  ASTE_RT-(CPU/GPU)

> """)

    if example == "0":
        example = "DRO_PO-CPU"

    elif example == "1":
        example = "DRO_PO-GPU"

    elif example == "2":
        example = "DRO_RT-CPU"

    elif example == "3":
        example = "DRO_RT-GPU"


    elif example == "4":
        example = "ASTE_PO-CPU"

    elif example == "5":
        example = "ASTE_PO-GPU"

    elif example == "6":
        example = "ASTE_RT-CPU"

    elif example == "7":
        example = "ASTE_RT-GPU"

    args = example.split("-")
    print("Running ex_{}.py on {}\n".format(args[0], args[1]))
    if args[0] == "DRO_PO":
        ex_DRO_PO(args[1])

    elif args[0] == "DRO_RT":
        ex_DRO_RT(args[1])

    elif args[0] == "ASTE_PO":
        ex_ASTE_PO(args[1])

    elif args[0] == "ASTE_RT":
        ex_ASTE_RT(args[1])

if __name__ == "__main__":
    select_example()
