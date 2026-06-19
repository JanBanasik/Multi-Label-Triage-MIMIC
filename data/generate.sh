#!/bin/bash

# Ustawienia
BATCH_COUNT=10        # Ile partii (10 partii x 5000 pacjentów = 50k)
PATIENTS_PER_BATCH=5000
JAR_FILE="synthea-with-dependencies.jar"

# Sprawdź czy plik jar istnieje
if [ ! -f "$JAR_FILE" ]; then
    echo "❌ Błąd: Nie znaleziono pliku $JAR_FILE w tym katalogu!"
    exit 1
fi

echo "🚀 Rozpoczynam generowanie danych Synthea..."
echo "📊 Cel: $BATCH_COUNT partii po $PATIENTS_PER_BATCH pacjentów."

# Pętla generująca
for i in $(seq 1 $BATCH_COUNT)
do
    echo "--------------------------------------------------"
    echo "🔄 Generowanie Partii #$i (Seed: $i)..."
    echo "--------------------------------------------------"
    
    # Tworzymy osobny folder dla każdej partii, np. ./data/batch_1
    OUTPUT_DIR="./data/batch_$i"
    
    # Uruchomienie Synthea
    # -s $i : Ustawia seed na numer partii (gwarancja unikalności między partiami)
    # --exporter.baseDirectory : Zapisuje wyniki do osobnego folderu
    java -jar $JAR_FILE \
        -p $PATIENTS_PER_BATCH \
        -s $i \
        --exporter.csv.export true \
        --exporter.fhir.export false \
        --exporter.baseDirectory "$OUTPUT_DIR"
        
    echo "✅ Partia #$i zakończona. Wyniki w: $OUTPUT_DIR/csv"
done

echo "--------------------------------------------------"
echo "🎉 ZAKOŃCZONO! Wygenerowano łącznie $((BATCH_COUNT * PATIENTS_PER_BATCH)) pacjentów."