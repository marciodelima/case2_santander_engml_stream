# Case 2 - APP Streaming - Databricks - F1rst
## DataMaster - Engenharia de ML - JAN/2022

### Objetivo
Receber dados de pacientes via streaming e indicar se os pacientes têm ou não a probabilidade de ter uma doença cardíaca. 

### Uso
Ferramenta on-line em streaming que recebe dados de exames de pacientes de laboratórios e armazena os resultados. Com os resultados, a ferramenta pode notificar o(s) médico(s) sobre o caso e proporcionar ao paciente um tratamento mais especializado e mais rápido possível. 

### Escopo
Para o alcance do objetivo acima, foi criado uma infra-estrutura em cloud (Azure) via script em Terraform para o processamento em streaming dos dados. Utilizado a plataforma Databricks, MLFLOW e o EventHub. Foi criado um notebook para o envio de dados simulando o envio por um laboratório e de um paciente. Após o envio dos dados, os mesmos são processados e registrados / gravados numa delta table do Databricks. 

### Fora de Escopo
Notificação dos resultados ao(s) médico(s). 

### Desenho

![Desenho de Solução](https://github.com/marciodelima/case2_santander_engml_stream/blob/main/desenho.png)
