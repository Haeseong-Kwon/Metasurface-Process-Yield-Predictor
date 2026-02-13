'use client';

import { useState } from 'react';
import { Card, Title, TextInput, NumberInput, Button, Grid, Col, Text } from '@tremor/react';
import { supabase } from '@/lib/supabase';
import { Save, Beaker } from 'lucide-react';

export default function RecipeForm() {
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({
        recipe_name: '',
        etching_time: 120,
        pressure: 50,
        temperature: 25,
        gas_flow_rate: 100,
    });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);

        try {
            const { error } = await supabase
                .from('process_runs')
                .insert([formData]);

            if (error) throw error;

            alert('Recipe successfully registered!');
            setFormData({
                recipe_name: '',
                etching_time: 120,
                pressure: 50,
                temperature: 25,
                gas_flow_rate: 100,
            });
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : 'Unknown error';
            console.error('Error saving recipe:', message);
            alert('Failed to save recipe: ' + message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm">
            <div className="flex items-center space-x-2 mb-6">
                <Beaker className="text-blue-400 w-6 h-6" />
                <Title className="text-slate-100">Recipe Management</Title>
            </div>
            <form onSubmit={handleSubmit} className="space-y-4">
                <Grid numItems={1} numItemsLg={2} className="gap-4">
                    <Col>
                        <Text className="text-slate-400 mb-1">Recipe Identifier</Text>
                        <TextInput
                            placeholder="e.g., Nano-Gate-V1"
                            value={formData.recipe_name}
                            onChange={(e) => setFormData({ ...formData, recipe_name: e.target.value })}
                            required
                            className="bg-slate-800 border-slate-700 text-slate-100"
                        />
                    </Col>
                    <Col>
                        <Text className="text-slate-400 mb-1">Etching Time (s)</Text>
                        <NumberInput
                            value={formData.etching_time}
                            onValueChange={(val) => setFormData({ ...formData, etching_time: val })}
                            required
                            className="bg-slate-800 border-slate-700 text-slate-100"
                        />
                    </Col>
                    <Col>
                        <Text className="text-slate-400 mb-1">Chamber Pressure (mTorr)</Text>
                        <NumberInput
                            value={formData.pressure}
                            onValueChange={(val) => setFormData({ ...formData, pressure: val })}
                            required
                            className="bg-slate-800 border-slate-700 text-slate-100"
                        />
                    </Col>
                    <Col>
                        <Text className="text-slate-400 mb-1">Substrate Temp (°C)</Text>
                        <NumberInput
                            value={formData.temperature}
                            onValueChange={(val) => setFormData({ ...formData, temperature: val })}
                            required
                            className="bg-slate-800 border-slate-700 text-slate-100"
                        />
                    </Col>
                    <Col numColSpanLg={2}>
                        <Text className="text-slate-400 mb-1">Gas Flow Rate (sccm)</Text>
                        <NumberInput
                            value={formData.gas_flow_rate}
                            onValueChange={(val) => setFormData({ ...formData, gas_flow_rate: val })}
                            required
                            className="bg-slate-800 border-slate-700 text-slate-100"
                        />
                    </Col>
                </Grid>
                <div className="mt-6 flex justify-end">
                    <Button
                        icon={Save}
                        loading={loading}
                        className="bg-blue-600 hover:bg-blue-500 border-none px-8 py-2 text-white font-medium transition-all"
                        type="submit"
                    >
                        Register Recipe
                    </Button>
                </div>
            </form>
        </Card>
    );
}
