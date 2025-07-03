import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pynfft
import time
from time import perf_counter



def golden_ratio_3d_sampling(m_max, M, phi1=0.4656, phi2=0.6823, visualize=False):
    """
    Gera pontos de amostragem 3D usando razões áureas ajustadas e interpolação
    
    Parâmetros:
    m_max : número máximo de amostras base
    M : número de pontos de interpolação entre amostras
    phi1, phi2 : razões áureas ajustadas
    visualize : se True, plota os pontos gerados
    
    Retorna:
    k_points : array (n_points, 3) com coordenadas no espaço k normalizadas [-π, π]
    labels : array (n_points,) com rótulos das pirâmides ('F', 'G', 'H')
    """
    # 1. Geração das amostras base usando razões áureas
    m_values = np.arange(1, m_max)
    beta = np.arccos(np.mod(m_values * phi1, 1))
    alpha = 2 * np.pi * np.mod(m_values * phi2, 1)
    
    # 2. Gerar pontos na esfera
    k_x1, k_y1, k_z1 = np.sin(beta) * np.cos(alpha), np.sin(beta) * np.sin(alpha), np.cos(beta)
    k_x2, k_y2, k_z2 = -k_x1, -k_y1, -k_z1  # Pontos opostos
    
    # 3. Normalização para o cubo [-1, 1]
    def normalize_to_cube(x, y, z):
        norm = np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)])
        return x / (norm + 1e-8), y / (norm + 1e-8), z / (norm + 1e-8)
    
    k_x1, k_y1, k_z1 = normalize_to_cube(k_x1, k_y1, k_z1)
    k_x2, k_y2, k_z2 = normalize_to_cube(k_x2, k_y2, k_z2)
    
    # 4. Escala para [-π, π]
    k_x1, k_y1, k_z1 = k_x1 * np.pi, k_y1 * np.pi, k_z1 * np.pi
    k_x2, k_y2, k_z2 = k_x2 * np.pi, k_y2 * np.pi, k_z2 * np.pi
    
    # 5. Interpolação entre pontos opostos
    I = np.linspace(0, 1, M)
    k_x = k_x1[:, None] + (k_x2 - k_x1)[:, None] * I
    k_y = k_y1[:, None] + (k_y2 - k_y1)[:, None] * I
    k_z = k_z1[:, None] + (k_z2 - k_z1)[:, None] * I
    
    # 6. Classificação em pirâmides
    def assign_to_pyramids(x, y, z):
        abs_vals = np.abs([x, y, z])
        max_idx = np.argmax(abs_vals)
        return ['F', 'G', 'H'][max_idx]
    
    # Formato final dos pontos e rótulos
    points = np.column_stack((k_x.flatten(), k_y.flatten(), k_z.flatten()))
    labels = np.array([assign_to_pyramids(x, y, z) for x, y, z in points])
    
    # Visualização (opcional)
    if visualize:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = {'F': 'red', 'G': 'green', 'H': 'blue'}
        for label in np.unique(labels):
            idx = labels == label
            ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2], 
                      color=colors[label], s=5, alpha=0.6, label=f'Pirâmide {label}')
        
        ax.set(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi), zlim=(-np.pi, np.pi),
               xlabel="k_x", ylabel="k_y", zlabel="k_z")
        ax.legend()
        plt.title(f"Amostragem 3D com Razões Áureas (N={m_max * M} pontos)")
        plt.tight_layout()
        nome_arquivo = f'Domínio_com_{qnt_raios}raios_{qnt_pontos}pontos.pdf'
        plt.savefig(nome_arquivo, bbox_inches='tight', dpi=300)
        plt.show()
    
    return points, labels

# Exemplo de uso:
qnt_raios = 150
qnt_pontos = 10
k_points, labels = golden_ratio_3d_sampling(m_max=qnt_raios, M=qnt_pontos, visualize=False)

k_points_normalized = k_points / (2 * np.pi)  # De [-π,π] para [-0.5,0.5]


#Usando pyNFFT

def reconstruct_3d(k_points, k_data, img_shape=(64, 64, 64)):
    """
    Reconstrução 3D a partir de dados não uniformes no k-space
    
    Parâmetros:
    k_points : array (n_points, 3) com coordenadas no k-space (normalizadas [-0.5, 0.5])
    k_data : array (n_points,) com valores complexos dos dados adquiridos
    img_shape : tupla com dimensões da imagem de saída
    
    Retorna:
    reconstructed_image : array 3D com a imagem reconstruída
    """
    # Verificar consistência dos dados
    assert k_points.shape[0] == k_data.shape[0], "Número de pontos e dados deve ser igual"
    assert k_points.shape[1] == 3, "Pontos devem ser 3D"
    
    # Inicializar plano NFFT 3D
    plan = pynfft.NFFT(img_shape, k_points.shape[0])
    
    # Configurar pontos de amostragem
    plan.x = k_points
    
    # Pré-computação
    plan.precompute()
    
    # Atribuir dados medidos
    plan.f = k_data
    
    # Reconstrução (operador adjunto)
    f_hat = plan.adjoint()
    
    # Ajuste de fase e magnitude
    reconstructed_image = np.abs(np.fft.ifftn(np.fft.ifftshift(f_hat)))
    
    return reconstructed_image

#Exemplo de uso:
# 1. Criar um phantom 3D simples (cubo)
img_shape = (64, 64, 64)
phantom = np.zeros(img_shape)
phantom[16:48, 16:48, 16:48] = 1.0

# Simular dados no k-space (como no seu exemplo anterior)
plan_forward = pynfft.NFFT(img_shape, k_points_normalized.shape[0])
plan_forward.x = k_points_normalized
plan_forward.precompute()
plan_forward.f_hat = np.fft.fftshift(np.fft.fftn(phantom))
k_data_simulated = plan_forward.trafo()

# Medição do tempo de reconstrução
start_time = time.time()
start = perf_counter()


# Reconstruir
reconstructed = reconstruct_3d(k_points_normalized, k_data_simulated, img_shape)

end_time = time.time()
reconstruction_time = end_time - start_time
elapsed = perf_counter() - start


print(f"Reconstrução concluída em {reconstruction_time:.2f} segundos")
print(f"Tamanho da imagem: {img_shape}")
print(f"Número de pontos k-space: {len(k_points_normalized)}")
print(f"Tempo de reconstrução: {elapsed:.4f} segundos (precisão de microssegundos)")



def calculate_metrics(original, reconstructed, reconstruction_time):
    """Calcula métricas de qualidade da reconstrução"""
    mse = np.mean((original - reconstructed)**2)
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(np.max(original) / (np.sqrt(mse) + 1e-10))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'PSNR': psnr,
        'Tempo (s)': reconstruction_time,
        'Pontos k-space': len(k_points_normalized),
        'Resolução': img_shape
    }

metrics = calculate_metrics(phantom, reconstructed, reconstruction_time)
for k, v in metrics.items():
    print(f"{k}: {v}")


# Visualização de cortes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

info_text = f"Raios: {qnt_raios}\nPontos por raio: {qnt_pontos}\nTotal pontos: {qnt_raios*qnt_pontos}"
fig.text(0.5, 0.95, info_text, ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

axes[0].imshow(reconstructed[img_shape[0]//2, :, :], cmap='gray')
axes[0].set_title('Corte Axial')
axes[1].imshow(reconstructed[:, img_shape[1]//2, :], cmap='gray')
axes[1].set_title('Corte Coronal')
axes[2].imshow(reconstructed[:, :, img_shape[2]//2], cmap='gray')
axes[2].set_title('Corte Sagital')
plt.tight_layout()


nome_arquivo = f'Reconstrucao_{qnt_raios}raios_{qnt_pontos}pontos.pdf'
#plt.savefig(nome_arquivo, bbox_inches='tight', dpi=300)
plt.show()
