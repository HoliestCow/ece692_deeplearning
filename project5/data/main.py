
import numpy as np
import glob
import h5py

def main():
    f = h5py.File('spectral_data.h5', 'w')
    # f['/'].create_dataset(
    # filelist = glob.glob('./integrations/bg_spectra_only*.npy')
    binno = 1024
    # use 0-3 for training, 4-5 for validation
    # 3 for testing i think would be good
    training_files = ['./integrations/bg_spectra_only_0.npy',
                      './integrations/bg_spectra_only_1.npy',
                      './integrations/bg_spectra_only_2.npy']
    testing_files = ['./integrations/bg_spectra_only_3.npy']
    validation_files = ['./integrations/bg_spectra_only_4.npy',
                        './integrations/bg_spectra_only_5.npy']

    # training_spectra = np.zeros((0, 1024))
    training_spectra = load_spectra(training_files, binno)
    # index = np.random.randint(training_spectra.shape[0], size=24 * 3600)
    # training_spectra = training_spectra[index, :]
    # np.save('./limited_training_spectra.npy', training_spectra)
    f.create_dataset('training_data', data=training_spectra, dtype=int, compression='gzip')
    training_spectra = []
    del training_spectra

    testing_spectra = load_spectra(testing_files, binno)
    # index = np.random.randint(testing_spectra.shape[0], size=12 * 3600)
    # testing_spectra = testing_spectra[index, :]
    # np.save('./limited_testing_spectra.npy', testing_spectra)
    f.create_dataset('testing_data', data=testing_spectra, dtype=int, compression='gzip')
    testing_spectra = []
    del testing_spectra

    validation_spectra = load_spectra(validation_files, binno)
    # index = np.random.randint(validation_spectra.shape[0], size=12 * 3600)
    # validation_spectra = validation_spectra[index, :]
    # np.save('./limited_validation_spectra.npy', validation_spectra)
    f.create_dataset('validation_data', data=validation_spectra, dtype=int, compression='gzip')
    validation_spectra = []
    del validation_spectra

    f.close()

    return


def load_spectra(filelist, binno):
    spectra = np.zeros((0, binno))
    for item in filelist:
        spectra = np.vstack((spectra, np.load(item)))
    return spectra

main()