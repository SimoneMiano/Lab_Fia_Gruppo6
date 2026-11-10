from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Cartella della prova da analizzare, relativa alla posizione di questo script
DATA_FOLDER = Path(__file__).resolve().parent.parent / "Gruppo_3_stud_1_N"

# Tipi di segnale disponibili:
#   file   -> nome del file CSV (senza estensione) nella cartella dei dati
#   prefix -> prefisso usato nelle chiavi delle feature (es. "accel_x_mean")
#   unit   -> unità di misura, usata nei grafici
SIGNAL_TYPES = {
    "acceleration": {"file": "Accelerometer", "prefix": "accel", "unit": "m/s²"},
    "gyroscope": {"file": "Gyroscope", "prefix": "gyro", "unit": "rad/s"},
}

# Ordine preferito dei canali (nei CSV l'ordine è z, y, x)
CHANNEL_ORDER = ["x", "y", "z", "qx", "qy", "qz", "qw", "roll", "pitch", "yaw"]


"""Carica un CSV e restituisce (seconds, dict canale -> array numpy)."""
def load_signal(file_path):
    data = np.genfromtxt(file_path, delimiter=",", names=True)
    seconds = data["seconds_elapsed"]
    columns = [c for c in data.dtype.names if c not in ("time", "seconds_elapsed")]
    columns.sort(key=lambda c: CHANNEL_ORDER.index(c) if c in CHANNEL_ORDER else -1)
    channels = {c: data[c] for c in columns}
    return seconds, channels


"""Carica il file e restituisce la media di ogni componente del segnale.

Esempio per signal_type="acceleration":
    {"accel_x_mean": ..., "accel_y_mean": ..., "accel_z_mean": ..., "accel_magnitude_mean": ...}
"""
def compute_features(file_path, signal_type):
    # Cerca il segnale tra quelli disponibili, altrimenti solleva un'eccezione
    if signal_type not in SIGNAL_TYPES:
        raise ValueError(
            f"Tipo '{signal_type}' non valido. Opzioni: {', '.join(SIGNAL_TYPES)}"
        )
    # Usa il prefisso appropriato per le chiavi del dizionario restituito
    prefix = SIGNAL_TYPES[signal_type]["prefix"]

    # Carica il file e restituisce il tempo (primo argomento, qui ignorato) e un dizionario canale -> array numpy.
    _, channels = load_signal(file_path)

    # calcola le features sul modulo del segnale (radice quadrata della somma dei quadrati dei canali x, y, z)
    # ed aggiungilo come canale "magnitude" al dizionario channels
    if all(c in channels for c in ("x", "y", "z")):
        magnitude = np.sqrt(channels["x"] ** 2 + channels["y"] ** 2 + channels["z"] ** 2)
        channels["magnitude"] = magnitude

    # Restituisce un dizionario con le medie dei canali, usando il prefisso appropriato.
    result = {}
    for component, values in channels.items():
        # aggiunge una chiave al dizionario result, con il nome del canale e la statistica calcolata
        # la chiave è una stringa, creata tramite f-string, es. "accel_x_mean"
        # il valore è la statistica calcolata tramite numpy, es. np.mean(values)
        result[f"{prefix}_{component}_mean"] = float(np.mean(values))

    return result


"""Visualizza i canali x, y, z e il modulo di un segnale, con asse del tempo condiviso."""
def plot_signal(file_path, signal_type):
    seconds, channels = load_signal(file_path)
    channels["magnitude"] = np.sqrt(channels["x"] ** 2 + channels["y"] ** 2 + channels["z"] ** 2)
    unit = SIGNAL_TYPES[signal_type]["unit"]

    # zoomando o spostandosi su un grafico (lente / croce nella toolbar della
    # finestra) anche gli altri si aggiornano.
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
    for ax, (label, signal), color in zip(
        axes, channels.items(), ("tab:red", "tab:green", "tab:blue", "black")
    ):
        ax.plot(seconds, signal, color=color, linewidth=0.8)
        ax.set_ylabel(f"{label} ({unit})")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("tempo (s)")
    fig.suptitle(f"{signal_type} - {DATA_FOLDER.name}")
    fig.tight_layout()


if __name__ == "__main__":
    for signal_type, info in SIGNAL_TYPES.items():
        file_path = DATA_FOLDER / f"{info['file']}.csv"

        # calcola e stampa le features
        feats = compute_features(file_path, signal_type)
        print(f"\n{signal_type}:")
        for name, value in feats.items():
            print(f"  {name}: {value:.4f}")

        # prepara il grafico del segnale
        plot_signal(file_path, signal_type)

    # apre le finestre dei grafici (chiuderle con la X per terminare il programma)
    plt.show()