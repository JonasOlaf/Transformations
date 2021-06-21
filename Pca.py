from sklearn.decomposition import PCA as skPCA
from sklearn.manifold import Isomap as skIsomap
from Helpers import unroll_db, inroll_db




class PCA():
    def __init__(self, train_db, n):
        self.skpca = skPCA(n_components=n)
        udb, _ = unroll_db(train_db)
        self.skpca.fit_transform(udb)

    def transform(self, test_db):
        udb, index = unroll_db(test_db)
        trans_udb = self.skpca.transform(udb)
        return inroll_db(trans_udb, index)

    def transform_single(self, probe):
        probe = probe.reshape(1,-1)
        probe = self.skpca.transform(probe)
        return probe.reshape(-1,1)

    def get_eigenvalues(self):
        print(f'Described variance: {sum(self.skpca.explained_vaiance_)}')
        return self.skpca.explained_vaiance_


class Isomap():
    def __init__(self, train_db, n, neighbors):
        self.skisomap = skIsomap(n_components=n, n_neighbors=neighbors)
        udb, index = unroll_db(train_db)
        rdb = self.skisomap.fit_transform(udb)
        self.db = inroll_db(rdb, index)

    def transform(self, test_db):
        udb, index = unroll_db(test_db)
        trans_udb = self.skisomap.transform(udb)
        return inroll_db(trans_udb, index)
