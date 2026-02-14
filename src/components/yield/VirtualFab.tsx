'use client';

import { useState, useMemo } from 'react';
import { Card, Title, Text, Grid, Col, Flex, Button } from '@tremor/react';
import { Beaker, Save, Download } from 'lucide-react';
import { YieldPrediction } from '@/types/yield';
import RiskIndicator from './RiskIndicator';

interface Props {
    onSave?: (prediction: YieldPrediction) => void;
    onExport?: (prediction: YieldPrediction) => void;
}

export default function VirtualFab({ onSave, onExport }: Props) {
    const [params, setParams] = useState({
        e_beam_dose: 450, // uC/cm2
        etching_time: 120, // s
        gas_flow: 100, // sccm
        pressure: 50, // mTorr
    });

    // Simulated AI Logic
    const prediction = useMemo(() => {
        let yield_rate = 95;
        const risks: string[] = [];

        // Dose affects yield
        if (params.e_beam_dose < 400) {
            yield_rate -= (400 - params.e_beam_dose) * 0.1;
            risks.push('E-beam dose too low: Under-exposure risk');
        } else if (params.e_beam_dose > 550) {
            yield_rate -= (params.e_beam_dose - 550) * 0.15;
            risks.push('E-beam dose too high: Proximity effect detected');
        }

        // Etching time affects yield
        if (params.etching_time > 180) {
            yield_rate -= (params.etching_time - 180) * 0.2;
            risks.push('Etching time exceeds limit: Over-etching likely');
        }

        // Pressure & Gas flow synergy
        if (params.pressure > 80 && params.gas_flow < 50) {
            yield_rate -= 10;
            risks.push('Low flow at high pressure: Non-uniform plasma distribution');
        }

        return {
            predicted_yield: Math.max(0, Math.min(100, yield_rate)),
            risk_factors: risks,
        };
    }, [params]);

    const currentPrediction: YieldPrediction = {
        ...params,
        ...prediction,
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <Col numColSpanLg={2}>
                <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm">
                    <div className="flex items-center space-x-2 mb-6">
                        <Beaker className="text-blue-400 w-6 h-6" />
                        <Title className="text-slate-100">Process Simulator Control</Title>
                    </div>

                    <div className="space-y-8">
                        <div>
                            <Flex>
                                <Text className="text-slate-400">E-beam Dose (uC/cm²)</Text>
                                <Text className="text-blue-400 font-mono font-bold">{params.e_beam_dose}</Text>
                            </Flex>
                            <input
                                type="range"
                                className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer mt-3 accent-blue-500"
                                value={params.e_beam_dose}
                                onChange={(e) => setParams({ ...params, e_beam_dose: parseInt(e.target.value) })}
                                min={200}
                                max={800}
                                step={10}
                            />
                        </div>

                        <div>
                            <Flex>
                                <Text className="text-slate-400">Etching Time (s)</Text>
                                <Text className="text-blue-400 font-mono font-bold">{params.etching_time}</Text>
                            </Flex>
                            <input
                                type="range"
                                className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer mt-3 accent-blue-500"
                                value={params.etching_time}
                                onChange={(e) => setParams({ ...params, etching_time: parseInt(e.target.value) })}
                                min={30}
                                max={300}
                                step={1}
                            />
                        </div>

                        <Grid numItems={1} numItemsSm={2} className="gap-8">
                            <div>
                                <Flex>
                                    <Text className="text-slate-400">Gas Flow (sccm)</Text>
                                    <Text className="text-blue-400 font-mono font-bold">{params.gas_flow}</Text>
                                </Flex>
                                <input
                                    type="range"
                                    className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer mt-3 accent-blue-500"
                                    value={params.gas_flow}
                                    onChange={(e) => setParams({ ...params, gas_flow: parseInt(e.target.value) })}
                                    min={10}
                                    max={200}
                                    step={5}
                                />
                            </div>
                            <div>
                                <Flex>
                                    <Text className="text-slate-400">Pressure (mTorr)</Text>
                                    <Text className="text-blue-400 font-mono font-bold">{params.pressure}</Text>
                                </Flex>
                                <input
                                    type="range"
                                    className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer mt-3 accent-blue-500"
                                    value={params.pressure}
                                    onChange={(e) => setParams({ ...params, pressure: parseInt(e.target.value) })}
                                    min={5}
                                    max={150}
                                    step={1}
                                />
                            </div>
                        </Grid>
                    </div>

                    <div className="mt-10 flex space-x-3 justify-end border-t border-slate-800 pt-6">
                        <Button
                            variant="secondary"
                            icon={Download}
                            onClick={() => onExport?.(currentPrediction)}
                            className="border-slate-700 text-slate-300 hover:bg-slate-800"
                        >
                            Export Report
                        </Button>
                        <Button
                            icon={Save}
                            onClick={() => onSave?.(currentPrediction)}
                            className="bg-blue-600 hover:bg-blue-500 border-none"
                        >
                            Save Prediction
                        </Button>
                    </div>
                </Card>
            </Col>

            <div className="space-y-8">
                <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm">
                    <Title className="text-slate-100 mb-2">Yield Prediction</Title>
                    <div className="flex flex-col items-center py-6">
                        <div className="relative flex items-center justify-center">
                            <svg className="w-48 h-48 transform -rotate-90">
                                <circle
                                    cx="96"
                                    cy="96"
                                    r="80"
                                    strokeWidth="12"
                                    stroke="currentColor"
                                    fill="transparent"
                                    className="text-slate-800"
                                />
                                <circle
                                    cx="96"
                                    cy="96"
                                    r="80"
                                    strokeWidth="12"
                                    strokeDasharray={502.4}
                                    strokeDashoffset={502.4 - (502.4 * currentPrediction.predicted_yield) / 100}
                                    strokeLinecap="round"
                                    stroke="currentColor"
                                    fill="transparent"
                                    className={`${currentPrediction.predicted_yield > 85 ? 'text-emerald-500' :
                                        currentPrediction.predicted_yield > 70 ? 'text-amber-500' : 'text-rose-500'
                                        } transition-all duration-500`}
                                />
                            </svg>
                            <div className="absolute flex flex-col items-center">
                                <span className="text-4xl font-bold font-mono text-white">{currentPrediction.predicted_yield.toFixed(1)}%</span>
                                <span className="text-xs uppercase tracking-widest text-slate-500 font-bold">Estimated</span>
                            </div>
                        </div>
                        <Text className="mt-4 text-slate-400 text-center">
                            Predicted yield based on current configuration
                        </Text>
                    </div>
                </Card>

                <RiskIndicator
                    riskFactors={currentPrediction.risk_factors}
                    predictedYield={currentPrediction.predicted_yield}
                />
            </div>
        </div>
    );
}
