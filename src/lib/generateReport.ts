import { jsPDF } from 'jspdf';
import { YieldPrediction } from '@/types/yield';

export const generatePredictionReport = (prediction: YieldPrediction) => {
    const doc = new jsPDF();

    // Header
    doc.setFillColor(15, 23, 42); // slate-900
    doc.rect(0, 0, 210, 40, 'F');

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(22);
    doc.text('Virtual Fab Simulation Report', 20, 25);

    // Meta Info
    doc.setTextColor(100, 116, 139); // slate-500
    doc.setFontSize(10);
    doc.text(`Generated on: ${new Date().toLocaleString()}`, 20, 35);

    // Section: Parameters
    doc.setTextColor(30, 41, 59); // slate-800
    doc.setFontSize(16);
    doc.text('Input Process Parameters', 20, 60);

    doc.setFontSize(12);
    const params = [
        ['E-beam Dose', `${prediction.e_beam_dose} uC/cm2`],
        ['Etching Time', `${prediction.etching_time} s`],
        ['Gas Flow Rate', `${prediction.gas_flow} sccm`],
        ['Chamber Pressure', `${prediction.pressure} mTorr`],
    ];

    let yPos = 75;
    params.forEach(([label, value]) => {
        doc.setTextColor(100, 116, 139);
        doc.text(label, 20, yPos);
        doc.setTextColor(30, 41, 59);
        doc.text(value, 100, yPos);
        yPos += 10;
    });

    // Section: Prediction
    doc.setDrawColor(226, 232, 240); // slate-200
    doc.line(20, yPos + 5, 190, yPos + 5);

    yPos += 20;
    doc.setFontSize(16);
    doc.text('AI Prediction Result', 20, yPos);

    yPos += 15;
    doc.setFontSize(28);
    const yieldColor = prediction.predicted_yield > 85 ? [16, 185, 129] : [244, 63, 94]; // emerald-500 or rose-500
    doc.setTextColor(yieldColor[0], yieldColor[1], yieldColor[2]);
    doc.text(`${prediction.predicted_yield.toFixed(2)}%`, 20, yPos);

    doc.setFontSize(10);
    doc.setTextColor(100, 116, 139);
    doc.text('Predicted Batch Yield Rate', 20, yPos + 8);

    // Section: Risk Factors
    if (prediction.risk_factors.length > 0) {
        yPos += 25;
        doc.setTextColor(244, 63, 94); // rose-500
        doc.setFontSize(14);
        doc.text('Risk Factors Identified', 20, yPos);

        doc.setFontSize(11);
        doc.setTextColor(71, 85, 105); // slate-600
        prediction.risk_factors.forEach((risk) => {
            yPos += 8;
            doc.text(`- ${risk}`, 25, yPos);
        });
    }

    // Footer
    doc.setFontSize(8);
    doc.setTextColor(148, 163, 184); // slate-400
    doc.text('Confidential - Fab Intelligence Systems', 105, 285, { align: 'center' });

    doc.save(`Simulation_Report_${Date.now()}.pdf`);
};
