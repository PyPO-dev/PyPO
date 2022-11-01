from examples.DRO_PSF_PO import ex_DRO_PO
from examples.DRO_PSF_RT import ex_DRO_RT
from examples.ASTE_PSF_PO import ex_ASTE_PO
from examples.ASTE_PSF_RT import ex_ASTE_RT

def select_example():
    example = input("""Welcome to the POPPy example interface!
Please select an example to run.

Possible options: DRO_PO-(CPU/GPU)
                  DRO_RT-CPU
                  ASTE_PO-(CPU/GPU)
                  ASTE_RT-CPU

> """)
    args = example.split("-")
    print("Running ex_{}.py on {}\n".format(args[0], args[1]))
    if args[0] == "DRO_PO":
        ex_DRO_PO(args[1])

    elif args[0] == "DRO_RT":
        ex_DRO_RT()

    elif args[0] == "ASTE_PO":
        ex_ASTE_PO(args[1])

    elif args[0] == "ASTE_RT":
        ex_ASTE_RT(args[1])

if __name__ == "__main__":
    select_example()
