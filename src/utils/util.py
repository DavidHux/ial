
def checkargs(args):
    from .. import const
    # print(configs)
    consts = const.configs

    # check dataset
    if args.dataset not in consts['datasets']:
        print('dataset error')
        exit(0)

    print(const.configs)

