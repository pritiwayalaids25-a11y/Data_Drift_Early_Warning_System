pipeline {
    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                bat 'python -m pip install -r requirements.txt'
            }
        }

        stage('Test') {
            steps {
                bat 'python -c "import pandas; import numpy; import scipy; print(\'Dependencies OK\')"'
            }
        }

        stage('Docker Build') {
            steps {
                bat 'docker build -t data-drift-warning:latest .'
            }
        }

        stage('Docker Image Verify') {
            steps {
                bat 'docker images data-drift-warning:latest'
            }
        }

        stage('Kubernetes Check') {
            steps {
                bat 'set KUBECONFIG=C:\\Users\\M_Dell\\.kube\\config && kubectl get nodes'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                bat 'set KUBECONFIG=C:\\Users\\M_Dell\\.kube\\config && kubectl apply -f k8s\\deployment.yaml'
                bat 'set KUBECONFIG=C:\\Users\\M_Dell\\.kube\\config && kubectl rollout status deployment/data-drift-app'
            }
        }

        stage('Verify Application') {
            steps {
                bat 'set KUBECONFIG=C:\\Users\\M_Dell\\.kube\\config && kubectl get pods'
            }
        }

        stage('Verify Service') {
            steps {
                bat 'set KUBECONFIG=C:\\Users\\M_Dell\\.kube\\config && kubectl get service data-drift-service'
            }
        }
    }
}