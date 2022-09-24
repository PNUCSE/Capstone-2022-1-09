# 기본 모듈 설치 (설치되어 있는 경우 생략 가능)
install.packages("BiocManager")
BiocManager::install("Rcpi", dependencies = c("Imports", "Enhances"), force=TRUE)
BiocManager::install("GenomeInfoDb", force=TRUE)
BiocManager::install("XVector", force=TRUE)
BiocManager::install("bit", force=TRUE)
BiocManager::install("Biobase", force=TRUE)
BiocManager::install("GO.db", force=TRUE)
BiocManager::install("rcdklibs", force=TRUE)

# 라이브러리 불러오기
library("Rcpi")

file = "경로명/data/protein_label_data.csv"
labelList = read.csv(file, header=TRUE)

rslt_df = c()
for(i in 1:nrow(labelList)){
	nowe = labelList[i, 1]
	nowl = labelList[i, 2]

	# 단백질 구조 추출
	# print(nowe)
	seqs = getProt(nowe, from = 'uniprot', type = 'aaseq')[[1]][[1]]
	# print(seqs)
	# 단백질 meta 정보 추출
	rslt = extractProtPAAC(seqs)

	trslt = t(rslt)  # 행/열 뒤집기
	rownames(trslt) = nowl  # 행 이름 지정

	rslt_df = rbind(rslt_df, trslt)
}

directory_path = '경로명/data'
setwd(directory_path)
write.csv(rslt_df, file='protein_data.csv', quote=FALSE)