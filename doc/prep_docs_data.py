import BPt as bp

def prep_data():

    # Save an up to data example1 dataset
    data = bp.read_csv('data/example1.csv')
    data.to_pickle('data/example1.dataset')