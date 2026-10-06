<?php

/*
 * SPDX-FileCopyrightText: 2026 Recognize contributors
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */
declare(strict_types=1);

use OCA\Recognize\Command\ExportAdaptiveFaces;
use OCA\Recognize\Db\FaceDetection;
use OCA\Recognize\Db\FaceDetectionMapper;
use OCP\IDBConnection;
use OCP\Server;
use Symfony\Component\Console\Tester\CommandTester;
use Test\TestCase;

/** @group DB */
class ExportAdaptiveFacesTest extends TestCase {

	public function testExportIsScopedAndDoesNotChangeDetection(): void {
		$db = Server::get(IDBConnection::class);
		$mapper = Server::get(FaceDetectionMapper::class);
		$user = 'adaptive-export-' . bin2hex(random_bytes(8));
		$db->beginTransaction();
		try {
			foreach ([$user, $user . '-other'] as $owner) {
				$detection = new FaceDetection();
				$detection->setUserId($owner);
				$detection->setFileId(123);
				$detection->setX(0.1);
				$detection->setY(0.2);
				$detection->setWidth(0.3);
				$detection->setHeight(0.4);
				$detection->setThreshold(0.5);
				$detection->setClusterId(null);
				$detection->setVector(array_merge([1.0], array_fill(0, 511, 0.0)));
				$mapper->insert($detection);
			}
			$before = $mapper->findByUserId($user)[0];
			$tester = new CommandTester(new ExportAdaptiveFaces($db));
			$this->assertSame(0, $tester->execute(['user' => $user, 'embedding-model' => 'synthetic-unit-v1']));
			$data = json_decode($tester->getDisplay(), true, 512, JSON_THROW_ON_ERROR);
			$this->assertSame('recognize-adaptive-snapshot-v1', $data['schema']);
			$this->assertCount(1, $data['faces']);
			$this->assertSame($user, $data['faces'][0]['user_id']);
			$this->assertSame($before->getId(), $data['faces'][0]['id']);
			$this->assertSame(0.5, $data['faces'][0]['threshold']);
			$this->assertNull($data['faces'][0]['cluster_id']);
			$this->assertCount(512, $data['faces'][0]['vector']);
			$this->assertArrayNotHasKey('face_vector', $data['faces'][0]);
			$after = $mapper->findByUserId($user)[0];
			$this->assertSame($before->toArray(), $after->toArray());
			$this->assertSame($before->getVector(), $after->getVector());
			$this->assertSame(0, $tester->execute(['user' => $user . "' OR 1=1 --", 'embedding-model' => 'synthetic-unit-v1']));
			$this->assertSame([], json_decode($tester->getDisplay(), true, 512, JSON_THROW_ON_ERROR)['faces']);
		} finally {
			$db->rollBack();
		}
	}

	public function testEmptyModelIsRejected(): void {
		$tester = new CommandTester(new ExportAdaptiveFaces(Server::get(IDBConnection::class)));
		$this->expectException(\InvalidArgumentException::class);
		$tester->execute(['user' => 'example', 'embedding-model' => ' ']);
	}
}
