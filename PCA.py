from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import kutuphane
 
giris, cikis, kisi_bilgisi =  kutuphane.dosya_oku('data/pd_speech_features.csv')


scaler = StandardScaler()
X = scaler.fit_transform(giris)

pca = PCA(n_components=50)
pca_x = pca.fit_transform(X)


accuracy, f1_skor = kutuphane.basari_hesaplaCV(pca_x, cikis,kisi_bilgisi)
print("pca basarisi = "+ str(accuracy) )


lda = LDA(n_components=2)
lda_x =lda.fit_transform(X,cikis)


accuracy, f1_skor = kutuphane.basari_hesapla(lda_x, cikis, kisi_bilgisi)
print("LDA basarisi = "+ str(accuracy) )
