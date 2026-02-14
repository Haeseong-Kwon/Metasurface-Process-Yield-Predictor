import { ProcessWithYield } from '@/types/yield';

export const exportProcessHistoryCSV = (history: ProcessWithYield[]) => {
    if (history.length === 0) return;

    const headers = [
        'Batch ID',
        'Date',
        'Recipe',
        'Etching Time (s)',
        'Pressure (mTorr)',
        'Temperature (C)',
        'Gas Flow (sccm)',
        'Yield Rate (%)',
        'Efficiency',
        'Defect Density',
    ];

    const rows = history.map((run) => [
        run.id,
        new Date(run.created_at).toISOString(),
        run.recipe_name,
        run.etching_time,
        run.pressure,
        run.temperature,
        run.gas_flow_rate,
        run.yield_results?.[0]?.yield_rate || 0,
        run.yield_results?.[0]?.efficiency || 0,
        run.yield_results?.[0]?.defect_density || 0,
    ]);

    const csvContent = [
        headers.join(','),
        ...rows.map((row) => row.join(',')),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `Process_History_Export_${Date.now()}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};
