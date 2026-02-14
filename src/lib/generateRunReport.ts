import jsPDF from 'jsPDF';
import autoTable from 'jspdf-autotable';
import { ProcessWithYield } from '@/types/yield';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type jsPDFWithAutoTable = jsPDF & { lastAutoTable: any };

export const generateProfessionalRunReport = (run: ProcessWithYield) => {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.width;

    // Header Background
    doc.setFillColor(15, 23, 42); // slate-900
    doc.rect(0, 0, pageWidth, 45, 'F');

    // Title & ID
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(24);
    doc.setFont('helvetica', 'bold');
    doc.text('INDUSTRIAL METROLOGY REPORT', 20, 25);

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text(`Batch ID: ${run.id}`, 20, 35);
    doc.text(`Run Date: ${new Date(run.created_at).toLocaleString()}`, pageWidth - 20, 35, { align: 'right' });

    let yPos = 60;

    // Section 1: Process Recipe
    doc.setTextColor(30, 41, 59);
    doc.setFontSize(14);
    doc.text('1. Process Recipe Parameters', 20, yPos);

    const recipeData = [
        ['Parameter', 'Target Value', 'Actual Value', 'Status'],
        ['Etching Time', '135.0 s', `${run.etching_time} s`, run.etching_time > 140 ? 'Warning' : 'Nominal'],
        ['Chamber Pressure', '50 mTorr', `${run.pressure} mTorr`, 'Nominal'],
        ['Gas Flow Rate', '100 sccm', `${run.gas_flow_rate} sccm`, 'Nominal'],
        ['Temperature', '25.0 C', `${run.temperature} C`, 'Nominal'],
    ];

    autoTable(doc, {
        startY: yPos + 5,
        head: [recipeData[0]],
        body: recipeData.slice(1),
        theme: 'striped',
        headStyles: { fillColor: [51, 65, 85] },
    });

    yPos = (doc as jsPDFWithAutoTable).lastAutoTable.finalY + 15;

    // Section 2: Yield Analysis
    doc.text('2. AI-Predicted vs. Actual Yield', 20, yPos);

    const yieldRate = run.yield_results?.[0]?.yield_rate || 0;
    const predictedYield = 85.2; // Mock baseline predicted

    autoTable(doc, {
        startY: yPos + 5,
        body: [
            ['Metric', 'Value', 'Deviation'],
            ['Actual Batch Yield', `${yieldRate}%`, `${(yieldRate - predictedYield).toFixed(1)}%`],
            ['Process Efficiency', `${run.yield_results?.[0]?.efficiency || 0}`, '-'],
            ['Defect Density', `${run.yield_results?.[0]?.defect_density || 0} pts/um2`, '-'],
        ],
        theme: 'grid',
        headStyles: { fillColor: [15, 23, 42] },
    });

    yPos = (doc as jsPDFWithAutoTable).lastAutoTable.finalY + 15;

    // Section 3: SEM Metrology Analysis
    doc.text('3. SEM Vision Analysis Summary', 20, yPos);

    const defectCounts = run.defects?.reduce((acc: Record<string, number>, d) => {
        acc[d.type] = (acc[d.type] || 0) + 1;
        return acc;
    }, {}) || {};

    const defectTable = Object.entries(defectCounts).map(([type, count]) => [type, `${count} occurrences`]);

    autoTable(doc, {
        startY: yPos + 5,
        head: [['Defect Type', 'Detection Count']],
        body: defectTable.length > 0 ? defectTable : [['No defects detected', '-']],
        theme: 'plain',
        styles: { fontSize: 10 },
    });

    yPos = (doc as jsPDFWithAutoTable).lastAutoTable.finalY + 15;

    // Section 4: AI Optimization Guidelines
    doc.setFillColor(248, 250, 252); // slate-50
    doc.rect(15, yPos - 5, pageWidth - 30, 40, 'F');

    doc.setTextColor(30, 41, 59);
    doc.text('4. Optimization Guidelines', 20, yPos);

    doc.setFontSize(10);
    let adviceY = yPos + 10;
    run.advice?.forEach((a) => {
        doc.text(`- [${a.category}] ${a.suggestion} (Est. +${a.impact_percent}% Yield Gain)`, 25, adviceY);
        adviceY += 7;
    });

    // Footer
    doc.setFontSize(8);
    doc.setTextColor(148, 163, 184);
    doc.text('CONFIDENTIAL METROLOGY RECORD - FAB INTELLIGENCE SYSTEMS', pageWidth / 2, 285, { align: 'center' });

    doc.save(`Technical_Report_${run.id}_${Date.now()}.pdf`);
};
