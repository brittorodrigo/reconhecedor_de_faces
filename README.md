# reconhecedor_de_faces
- Exemplos de como executar o programa:
   python face_recognizer.py bdangelina 5 1
   python face_recognizer.py bdangelina 5 2

- O programa recebe como parâmetros:
	- o nome do diretório com as fotos  da face da pessoa que serão usadas
para treinar o classificador; 
	- a quantidade de fotos da face da pessoa no diretório de teste;
	- e o ID do algortimo de reconhecimento de face a ser usado (1 para LBP e 2 para Eigenfaces) .
- O diretótiro "bdtest" possui fotos de diferentes pessoas (que foram treinadas ou não), e o classificador
deve identificar nessas fotos , com um determinado nível de confiança, uma determinada face que foi usada 
para treiná-lo.
- O programa escreve na saída (diretório das fotos da pessoa a ser identificada) um arquivo .txt contendo 
a taxa de acertos e erros


