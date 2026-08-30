
export async function fetchInitialData() {
    try {
        const [modelsRes, diseaseRes] = await Promise.all([
            fetch('/api/models'),
            fetch('/api/disease-info')
        ]);
        
        let modelsData = {};
        let diseaseInfo = {};
        
        if (modelsRes.ok) {
            const data = await modelsRes.json();
            data.models.forEach(m => modelsData[m.id] = m);
        }
        if (diseaseRes.ok) {
            diseaseInfo = await diseaseRes.json();
        }
        return { modelsData, diseaseInfo };
    } catch (e) {
        console.error("Failed to fetch initial data", e);
        return null;
    }
}
